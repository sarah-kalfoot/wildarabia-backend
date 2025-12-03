import os
import io
import base64
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
import torch.nn.functional as F
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =============== CONFIG ===============

PRIMARY_MODEL_ID = "Sarahkalfoot/WildArabia"
FALLBACK_MODEL_ID = "google/vit-base-patch16-224"

# نستخدم نسخة الـ Instruct من كوين 1.5B عن طريق HuggingFace API
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
HF_API_KEY = os.environ.get("HF_API_KEY")
QWEN_AVAILABLE = HF_API_KEY is not None

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, "rag", "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "rag", "docs")

CONFIDENCE_THRESHOLD = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Saudi Wildlife AI API with RAG + HF Qwen")

# =============== REQUEST / RESPONSE SCHEMAS ===============

class AnalyzeRequest(BaseModel):
    image_base64: str
    question: Optional[str] = None

class ClassificationResult(BaseModel):
    model_used: str
    label: str
    confidence: float
    top3: List[Dict[str, Any]]

class AnalyzeResponse(BaseModel):
    classification: ClassificationResult
    rag_snippets: List[str]
    analysis: str

# =============== GLOBALS ===============

primary_processor = None
primary_model = None
primary_labels: List[str] = []

fallback_processor = None
fallback_model = None

embeddings = None
vectordb = None

# =============== HELPER: BASE64 → PIL IMAGE ===============

def decode_base64_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        image_data = base64.b64decode(b64_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

# =============== LOAD PRIMARY MODEL (WildArabia) ===============

def load_primary_model():
    global primary_processor, primary_model, primary_labels

    try:
        primary_processor = AutoImageProcessor.from_pretrained(PRIMARY_MODEL_ID)
        primary_model = AutoModelForImageClassification.from_pretrained(
            PRIMARY_MODEL_ID
        ).to(device)
        primary_model.eval()

        if hasattr(primary_model.config, "id2label") and primary_model.config.id2label:
            id2label = primary_model.config.id2label
            primary_labels = [id2label[str(i)] for i in range(len(id2label))]
        else:
            num_labels = primary_model.config.num_labels
            primary_labels = [f"class_{i}" for i in range(num_labels)]

        print("Primary model loaded from HF:", PRIMARY_MODEL_ID)

    except Exception as e:
        print("Error loading primary model:", e)
        raise RuntimeError("Failed to load PRIMARY_MODEL. Check model repo or code.")

# =============== LOAD FALLBACK MODEL (google/vit-base...) ===============

def load_fallback_model():
    global fallback_processor, fallback_model

    try:
        fallback_processor = AutoImageProcessor.from_pretrained(FALLBACK_MODEL_ID)
        fallback_model = AutoModelForImageClassification.from_pretrained(
            FALLBACK_MODEL_ID
        ).to(device)
        fallback_model.eval()
        print("Fallback model loaded:", FALLBACK_MODEL_ID)
    except Exception as e:
        print("Error loading fallback model:", e)
        fallback_processor = None
        fallback_model = None

# =============== CLASSIFICATION FUNCS ===============

def classify_with_primary(image: Image.Image) -> ClassificationResult:
    if primary_model is None or primary_processor is None:
        raise RuntimeError("Primary model not loaded")

    inputs = primary_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = primary_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]

    top_probs, top_indices = torch.topk(probs, k=3)
    top_probs = top_probs.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    top3 = []
    for p, idx in zip(top_probs, top_indices):
        label = primary_labels[idx] if idx < len(primary_labels) else f"class_{idx}"
        top3.append({"label": label, "confidence": float(p)})

    best_label = top3[0]["label"]
    best_conf = top3[0]["confidence"]

    return ClassificationResult(
        model_used="primary",
        label=best_label,
        confidence=best_conf,
        top3=top3,
    )


def classify_with_fallback(image: Image.Image) -> ClassificationResult:
    if fallback_model is None or fallback_processor is None:
        raise RuntimeError("Fallback model not loaded")

    inputs = fallback_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = fallback_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]

    top_probs, top_indices = torch.topk(probs, k=3)
    top_probs = top_probs.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    id2label = fallback_model.config.id2label

    top3 = []
    for p, idx in zip(top_probs, top_indices):
        label = id2label.get(str(idx), f"class_{idx}")
        top3.append({"label": label, "confidence": float(p)})

    best_label = top3[0]["label"]
    best_conf = top3[0]["confidence"]

    return ClassificationResult(
        model_used="fallback",
        label=best_label,
        confidence=best_conf,
        top3=top3,
    )


def unified_classification(image: Image.Image) -> ClassificationResult:
    primary_result = classify_with_primary(image)

    if primary_result.confidence >= CONFIDENCE_THRESHOLD:
        return primary_result

    if fallback_model is not None and fallback_processor is not None:
        fallback_result = classify_with_fallback(image)
        if fallback_result.confidence > primary_result.confidence:
            return fallback_result

    return primary_result

# =============== RAG: LOAD / BUILD VECTORSTORE ===============

def setup_embeddings_and_vectordb():
    global embeddings, vectordb

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

    # لو فيه Chroma جاهزة على الديسك
    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        print("Loaded existing Chroma DB from:", CHROMA_DIR)
        return

    # لو ما فيه، نبنيها من الـ PDFs
    if os.path.isdir(DOCS_DIR):
        pdf_files = [
            os.path.join(DOCS_DIR, f)
            for f in os.listdir(DOCS_DIR)
            if f.lower().endswith(".pdf")
        ]

        docs = []
        for pdf in pdf_files:
            if not os.path.exists(pdf):
                continue
            loader = UnstructuredPDFLoader(pdf)
            docs.extend(loader.load())

        if not docs:
            print("No docs found for RAG, vectordb will be None")
            vectordb = None
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = splitter.split_documents(docs)

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
        vectordb.persist()
        print("Built new Chroma DB at:", CHROMA_DIR)
    else:
        vectordb = None
        print("DOCS_DIR not found, vectordb will be None")


def rag_search(query: str, k: int = 5) -> List[str]:
    if vectordb is None:
        return []

    docs = vectordb.similarity_search(query, k=k)
    snippets = []
    for d in docs:
        text = d.page_content
        if len(text) > 500:
            text = text[:500] + "..."
        snippets.append(text)
    return snippets

# =============== HF QWEN (REMOTE LLM) ===============

def load_qwen():
    # حالياً ما نحمل مودل محلي، بس نطبع لو التوكن موجود
    if QWEN_AVAILABLE:
        print("HF_API_KEY found. Using remote Qwen via HuggingFace Inference API.")
    else:
        print("HF_API_KEY not set. Qwen analysis will fall back to simple text.")


def generate_analysis_with_qwen(
    classification: ClassificationResult,
    rag_snippets: List[str],
    question: Optional[str] = None,
) -> str:

    if not QWEN_AVAILABLE:
        return (
            f"Predicted species: {classification.label} "
            f"(confidence {classification.confidence:.2f}).\n"
            "Detailed LLM analysis is disabled because HF_API_KEY is not configured."
        )

    # نبني البرومبت
    prompt = f"""
You are an assistant for Saudi wildlife species.

Predicted species: {classification.label}
Model used: {classification.model_used}
Confidence: {classification.confidence:.2f}

Relevant information from reference documents:
"""
    for i, snip in enumerate(rag_snippets):
        prompt += f"\n[Doc {i+1}] {snip}\n"

    if question:
        prompt += f"\nUser question: {question}\n"

    prompt += """
Write a clear, factual, concise answer about this species, suitable for non-experts.
If possible, mention:
- Short description of the species
- Habitat and distribution (especially in Saudi Arabia / Arabian Peninsula)
- Conservation status if available
- Any interesting facts from the documents
"""

    url = f"https://api-inference.huggingface.co/models/{QWEN_MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # HF ممكن يرجع list أو dict
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            if isinstance(item, dict) and "generated_text" in item:
                return item["generated_text"]
            return str(item)
        elif isinstance(data, dict):
            # أحياناً يرجع {"error": "..."}
            if "generated_text" in data:
                return data["generated_text"]
            if "error" in data:
                return f"Qwen HF API error: {data['error']}"
            return str(data)

        return str(data)

    except Exception as e:
        return (
            f"Predicted species: {classification.label} "
            f"(confidence {classification.confidence:.2f}).\n"
            f"LLM call failed with error: {e}"
        )

# =============== MAIN PIPELINE ===============

def analyze_image_pipeline(
    image: Image.Image, question: Optional[str] = None
) -> AnalyzeResponse:
    classification = unified_classification(image)
    rag_snippets = rag_search(classification.label, k=5)
    analysis_text = generate_analysis_with_qwen(
        classification=classification,
        rag_snippets=rag_snippets,
        question=question,
    )

    return AnalyzeResponse(
        classification=classification,
        rag_snippets=rag_snippets,
        analysis=analysis_text,
    )

# =============== FASTAPI ENDPOINTS ===============

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        image = decode_base64_image(req.image_base64)
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    try:
        result = analyze_image_pipeline(image, question=req.question)
        return result
    except Exception as e:
        print("Error in analyze:", e)
        raise HTTPException(status_code=500, detail="Internal server error")

# =============== STARTUP HOOK ===============

@app.on_event("startup")
def on_startup():
    print("Loading models and RAG...")
    load_primary_model()
    load_fallback_model()
    setup_embeddings_and_vectordb()
    load_qwen()
    print("Startup complete")

# للـ local run:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
