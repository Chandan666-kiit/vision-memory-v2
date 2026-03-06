"""
main.py  —  Vision & Memory Pipeline
Uses DeepFace (no C++ compilation) instead of face_recognition/dlib.
"""
import base64
import io
import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import faiss
from PIL import Image
from pymongo import MongoClient
from openai import OpenAI
from deepface import DeepFace
import streamlit as st

DIM             = 128   # DeepFace Facenet embeddings are 128-d
FAISS_THRESHOLD = 20.0  # Facenet L2 distance: same person ~5-15, diff person >20
TOP_K           = 3
MODEL_NAME      = "Facenet"   # fast + accurate, 128-d embeddings


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _openai():
    return OpenAI()


@st.cache_resource
def _get_col():
    uri = os.environ.get(
        "MONGO_URI",
        "mongodb+srv://chandansrinethvickey_db_user:test1234@cluster0.5cahbo9.mongodb.net/"
        "?retryWrites=true&w=majority&appName=Cluster0",
    )
    return MongoClient(uri)["vision_memory"]["people"]


@st.cache_resource
def _get_faiss() -> dict:
    col = _get_col()
    col.delete_many({"$or": [
        {"name": {"$in": ["", "unknown", None]}},
        {"embedding": {"$exists": False}},
        {"embedding": []},
    ]})
    index = faiss.IndexFlatL2(DIM)
    ids: list[str] = []
    for doc in col.find({"embedding": {"$exists": True},
                         "name": {"$nin": ["", None, "unknown"]}}):
        vec = np.array(doc["embedding"], dtype="float32")
        if len(vec) == DIM:
            index.add(vec.reshape(1, -1))
            ids.append(doc.get("doc_id") or str(doc["_id"]))
    print(f"[main] FAISS ready — {index.ntotal} profile(s).")
    return {"index": index, "ids": ids}


def get_people_col():
    return _get_col()


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING  (DeepFace Facenet — no dlib, no compilation)
# ─────────────────────────────────────────────────────────────────────────────

def _get_embedding(image_path: str) -> tuple[list, str]:
    """
    Returns (embedding, face_b64) or ([], '') if no face found.
    face_b64 is the cropped face as base64 JPEG for GPT-4o comparison.
    """
    try:
        result = DeepFace.represent(
            img_path    = image_path,
            model_name  = MODEL_NAME,
            enforce_detection = True,
            detector_backend  = "opencv",
        )
        if not result:
            return [], ""

        embedding  = result[0]["embedding"]           # 128-d list
        facial_area = result[0].get("facial_area", {})

        # Crop the face region for GPT-4o
        face_b64 = _crop_face(image_path, facial_area)
        return embedding, face_b64

    except Exception as e:
        print(f"[main] DeepFace error: {e}")
        return [], ""


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _crop_face(image_path: str, area: dict) -> str:
    """Crop face from image using DeepFace facial_area dict."""
    try:
        img = Image.open(image_path).convert("RGB")
        x = area.get("x", 0)
        y = area.get("y", 0)
        w = area.get("w", img.width)
        h = area.get("h", img.height)
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.25)
        left   = max(0, x - pad_x)
        top    = max(0, y - pad_y)
        right  = min(img.width,  x + w + pad_x)
        bottom = min(img.height, y + h + pad_y)
        face   = img.crop((left, top, right, bottom)).resize((224, 224))
        buf    = io.BytesIO()
        face.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[main] crop error: {e}")
        # Fall back to full image resized
        img  = Image.open(image_path).convert("RGB").resize((224, 224))
        buf  = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()


def _full_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o FACE COMPARISON  —  GOLDEN RULE
# ─────────────────────────────────────────────────────────────────────────────

def _same_person(live_b64: str, stored_b64: str) -> bool:
    """Ask GPT-4o: are these two face images the same person?"""
    try:
        res = _openai().chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    "You are a face verification system.\n"
                    "Image 1 = NEW photo.  Image 2 = STORED reference.\n\n"
                    "Are these the SAME person?\n"
                    "Compare: face shape, eyes, nose, mouth, jawline, skin tone.\n"
                    "IGNORE completely: hair colour/style, glasses, lighting, "
                    "expression, age difference, background, camera angle.\n"
                    "These may be taken years apart or in very different conditions — "
                    "focus only on permanent facial bone structure.\n"
                    "When in doubt, lean towards TRUE — a human verifier will "
                    "confirm if needed.\n\n"
                    'JSON only: {"same_person": true/false, '
                    '"confidence": "high/medium/low", "reason": "one sentence"}'
                )},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{live_b64}",
                               "detail": "high"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{stored_b64}",
                               "detail": "high"}},
            ]}],
            response_format={"type": "json_object"},
            max_tokens=150,
        )
        data       = json.loads(res.choices[0].message.content)
        same       = bool(data.get("same_person", False))
        confidence = data.get("confidence", "low")
        print(f"[main] GPT-4o → same={same} conf={confidence} | {data.get('reason','')}")
        # Accept high OR medium OR low confidence matches — FAISS already pre-filtered
        return same
    except Exception as e:
        print(f"[main] GPT-4o error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTE DETECTION  (GPT-4o-mini)
# ─────────────────────────────────────────────────────────────────────────────

def _vj(img_b64: str, prompt: str) -> dict:
    try:
        res = _openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text",  "text": prompt + "\nJSON only. No markdown."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]}],
            response_format={"type": "json_object"},
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {}


def _detect_attrs(img_b64: str) -> dict:
    return {
        "glasses":      bool(_vj(img_b64, '{"glasses": true/false} wearing glasses?').get("glasses", False)),
        "emotion":      _vj(img_b64, '{"emotion": "..."} primary emotion?').get("emotion", "neutral"),
        "age_estimate": _vj(img_b64, '{"age_estimate": "25-30"} age range?').get("age_estimate", "unknown"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECOGNITION CORE
# ─────────────────────────────────────────────────────────────────────────────

def _recognise(embedding: list, live_face_b64: str) -> tuple[str, str]:
    state = _get_faiss()
    index = state["index"]
    ids   = state["ids"]

    if index.ntotal == 0:
        print("[main] FAISS empty — no profiles yet.")
        return "", ""

    k    = min(TOP_K, index.ntotal)
    vec  = np.array(embedding, dtype="float32").reshape(1, -1)
    D, I = index.search(vec, k)

    candidates = []
    for dist, pos in zip(D[0].tolist(), I[0].tolist()):
        dist, pos = float(dist), int(pos)
        if dist <= FAISS_THRESHOLD and pos < len(ids):
            candidates.append((dist, ids[pos]))
            print(f"[main] FAISS candidate dist={dist:.4f}")

    if not candidates:
        print("[main] No FAISS candidates — unknown person.")
        return "", ""

    candidates.sort(key=lambda x: x[0])

    for dist, doc_id in candidates:
        doc = _get_col().find_one({"doc_id": doc_id})
        if not doc:
            continue
        stored_name = (doc.get("name") or "").strip()
        stored_face = doc.get("face_b64", "")
        if not stored_name or not stored_face:
            continue
        if _same_person(live_face_b64, stored_face):
            print(f"[main] ✅ Confirmed: '{stored_name}'")
            return stored_name, doc_id
        print(f"[main] ❌ Rejected: '{stored_name}'")

    print("[main] All candidates rejected — unknown.")
    return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# GREETING
# ─────────────────────────────────────────────────────────────────────────────

def _greeting(name: str, emotion: str, age: str, glasses: bool, is_new: bool) -> str:
    parts = [f"Hello, {name}! 👋"]
    if emotion and emotion not in ("", "unknown", "neutral"):
        parts.append(f"You look {emotion} today.")
    if age and age not in ("", "unknown"):
        parts.append(f"You appear to be around {age} years old.")
    if glasses:
        parts.append("I see you're wearing glasses.")
    parts.append("Nice to meet you — I'll remember you! ✨" if is_new else "Welcome back! 🎉")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyse_image(image_path: str) -> dict:
    out = {
        "person": False, "embedding": [], "face_b64": "",
        "glasses": False, "emotion": "unknown", "age_estimate": "unknown",
        "need_name": False, "person_name": "", "greeting_message": "",
    }

    embedding, face_b64 = _get_embedding(image_path)

    if not embedding:
        out["greeting_message"] = "No face detected in the image."
        print("[main] No face found.")
        return out

    out["person"]    = True
    out["embedding"] = embedding
    out["face_b64"]  = face_b64
    print(f"[main] Face found. FAISS has {_get_faiss()['index'].ntotal} profile(s).")

    out.update(_detect_attrs(_full_b64(image_path)))

    name, doc_id = _recognise(out["embedding"], out["face_b64"])

    if not name:
        out["need_name"] = True
        return out

    _get_col().update_one(
        {"doc_id": doc_id},
        {"$set": {"emotion": out["emotion"],
                  "last_seen": datetime.now(timezone.utc)}},
    )
    out["person_name"]      = name
    out["greeting_message"] = _greeting(
        name, out["emotion"], out["age_estimate"], out["glasses"], is_new=False)
    return out


def register_and_greet(
    image_path: str, embedding: list, face_b64: str,
    name: str, glasses: bool, emotion: str, age_estimate: str,
) -> dict:
    name = (name or "").strip()
    if not name or name.lower() == "unknown":
        return {"error": "Please enter a valid name."}

    doc_id = str(uuid.uuid4())
    doc = {
        "doc_id":     doc_id,
        "name":       name,
        "embedding":  embedding,
        "face_b64":   face_b64,
        "glasses":    glasses,
        "emotion":    emotion,
        "age":        age_estimate,
        "created_at": datetime.now(timezone.utc),
        "last_seen":  datetime.now(timezone.utc),
    }
    _get_col().insert_one(doc)

    state = _get_faiss()
    vec   = np.array(embedding, dtype="float32").reshape(1, -1)
    state["index"].add(vec)
    state["ids"].append(doc_id)

    print(f"[main] ✅ Saved '{name}' FAISS total={state['index'].ntotal}")

    return {
        "person": True, "need_name": False,
        "person_name": name, "glasses": glasses,
        "emotion": emotion, "age_estimate": age_estimate,
        "greeting_message": _greeting(
            name, emotion, age_estimate, glasses, is_new=True),
    }
