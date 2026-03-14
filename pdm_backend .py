# pdm_backend.py
# Backend API بدون Gradio – مبني من كود final_RAG_-kym.ipynb

import os
import textwrap
from pathlib import Path
from datetime import datetime
import uuid
import csv
import json

import faiss
from flask import Flask, request, jsonify
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# =========[ إعداد مفتاح Gemini ]=========

# يفضل تاخده من متغير بيئة في السيرفر
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBxP6oa2xXqiCbnfV1dcBSa3ew568bgkhg")

if not GEMINI_API_KEY:
    raise RuntimeError("⚠️ رجاءً حط مفتاح Gemini في المتغير GEMINI_API_KEY.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL_NAME = "gemini-2.5-flash"

# مسار ملف الـ PDF
PDF_PATH = Path("factory_qa.pdf")

# لوج التفاعلات
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_CSV_PATH = LOG_DIR / "pdm_chat_logs.csv"

# =========[ 1) قراءة ملف الـ PDF وتقسيمه لبلوكات سؤال/إجابة ]=========

def load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            pages.append(txt)
    return "\n".join(pages)

def split_qa_blocks(text: str):
    blocks = []
    current_role = None
    current_q = None
    current_a_lines = []

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in lines:
        if line.startswith("[ROLE]"):
            if current_role and current_q and current_a_lines:
                blocks.append(
                    {
                        "role": current_role,
                        "question": current_q,
                        "answer": "\n".join(current_a_lines).strip(),
                    }
                )
            current_role = line.replace("[ROLE]", "").strip()
            current_q = None
            current_a_lines = []
        elif line.startswith("Q:"):
            current_q = line[2:].strip()
            current_a_lines = []
        elif line.startswith("A:"):
            current_a_lines.append(line[2:].strip())
        else:
            if current_a_lines is not None:
                current_a_lines.append(line)

    if current_role and current_q and current_a_lines:
        blocks.append(
            {
                "role": current_role,
                "question": current_q,
                "answer": "\n".join(current_a_lines).strip(),
            }
        )

    return blocks

raw_text = load_pdf_text(PDF_PATH)
qa_blocks = split_qa_blocks(raw_text)

# =========[ 2) بناء الـ embeddings و FAISS index ]=========

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

corpus_texts = []
for b in qa_blocks:
    corpus_texts.append(
        f"ROLE: {b['role']}\nQUESTION: {b['question']}\nANSWER: {b['answer']}"
    )

embeddings = embed_model.encode(
    corpus_texts, convert_to_numpy=True, show_progress_bar=False
)

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# =========[ 3) دوال RAG والـ telemetry ]=========

def retrieve_context(query: str, top_k: int = 3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    return [corpus_texts[idx] for idx in I[0]]

SAFETY_ANOMALY_THRESHOLD = 0.8  # موجود في النوتبوك للأمان

def build_telemetry(state: str, vibration: float | None,
                    temperature: float | None, current: float | None):
    latest_readings = {
        "vibration": vibration,
        "temperature": temperature,
        "current": current,
    }
    return {
        "machine_id": "M-03",
        "state": state,
        "rul_cycles": None,
        "latest_readings": latest_readings,
        "anomaly_score": None,
    }

def apply_safety_policy(telemetry: dict | None, user_role: str):
    if telemetry is None:
        return "", False

    state = telemetry.get("state")

    if state == "Failing":
        msg = (
            "تحذير مهم: حالة الماكينة حالياً (Failing / عطل خطير).\n"
            "أوقف الماكينة فوراً ولا تحاول إعادة التشغيل، وتواصل مع مهندس الصيانة.\n\n"
        )
        return msg, True

    if state == "Healthy":
        msg = (
            "ملاحظة: حالة الماكينة المدخلة Healthy، "
            "لكن سيتم تحليل سؤالك وربطه بسجل الأعطال.\n\n"
        )
        return msg, False

    if state == "Degrading":
        msg = (
            "تنبيه: الماكينة في حالة Degrading، يفضّل التخطيط لصيانة قريبة.\n\n"
        )
        return msg, False

    return "", False

def build_system_instruction(user_role: str) -> str:
    base = (
        "أنت مساعد ذكي لصيانة تنبؤية في مصنع صناعي.\n"
        "استخدم سياق الدليل الفني وبيانات الماكينة.\n"
        "تجنّب اختراع معلومات غير موجودة.\n"
    )

    if user_role == "TECHNICIAN":
        role_part = (
            "المستخدم فني صيانة. اشرح الخطوات ببساطة وبدون تعقيد.\n"
        )
    elif user_role == "ENGINEER":
        role_part = (
            "المستخدم مهندس صيانة. يمكنك استخدام تفاصيل فنية أعمق.\n"
        )
    elif user_role == "MANAGER":
        role_part = (
            "المستخدم مدير. ركّز على الملخصات والتأثير على الإنتاج.\n"
        )
    else:
        role_part = "تعامل مع المستخدم كمستخدم في مجال الصيانة.\n"

    style_part = (
        "اكتب بالعربية الفصحى المبسطة، في شكل نقاط وخطوات عملية.\n"
    )

    return base + "\n" + role_part + style_part

def build_prompt(user_message: str, user_role: str,
                 telemetry: dict | None, contexts: list[str]):
    system_instruction = build_system_instruction(user_role)
    context_text = "\n\n---\n\n".join(contexts) if contexts else "لا يوجد سياق متاح."

    telemetry_text = ""
    if telemetry is not None:
        r = telemetry.get("latest_readings", {})
        telemetry_text = (
            "بيانات الماكينة:\n"
            f"- الحالة: {telemetry.get('state')}\n"
            f"- الاهتزاز: {r.get('vibration')}\n"
            f"- الحرارة: {r.get('temperature')}\n"
            f"- التيار: {r.get('current')}\n\n"
        )

    safety_msg, block_actions = apply_safety_policy(telemetry, user_role)

    user_part = f"سؤال المستخدم:\n{user_message}\n\n"
    if block_actions:
        user_part += (
            "بسبب أن الحالة Failing، لا تعطي خطوات تشغيل، فقط وضّح المخاطر "
            "وخطوات التأمين والتواصل مع المهندس.\n"
        )

    prompt = (
        f"{system_instruction}\n\n"
        f"{telemetry_text}"
        "سياق من الدليل:\n"
        f"{context_text}\n\n"
        f"{safety_msg}"
        f"{user_part}"
        "أعطِ الإجابة الآن."
    )

    return prompt, safety_msg, block_actions

def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("مفيش GEMINI_API_KEY.")
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", "") or "").strip()

def generate_answer(user_message: str, user_role: str, telemetry: dict | None):
    contexts = retrieve_context(user_message, top_k=3)
    prompt, safety_msg, blocked = build_prompt(
        user_message, user_role, telemetry, contexts
    )
    answer = call_gemini(prompt)
    answer = textwrap.fill(answer, width=100)
    return answer, contexts, safety_msg, blocked

# =========[ 4) Logging في CSV ]=========

if not LOG_CSV_PATH.exists():
    with open(LOG_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "session_id",
                "user_role",
                "machine_state",
                "vibration",
                "temperature",
                "current",
                "user_message",
                "model_answer",
                "safety_blocked",
            ]
        )

def log_interaction_to_csv(
    session_id: str,
    user_role: str,
    telemetry: dict | None,
    user_message: str,
    model_answer: str,
    safety_blocked: bool,
):
    state = telemetry.get("state") if telemetry else None
    readings = telemetry.get("latest_readings", {}) if telemetry else {}
    vibration = readings.get("vibration")
    temperature = readings.get("temperature")
    current = readings.get("current")

    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                session_id,
                user_role,
                state,
                vibration,
                temperature,
                current,
                user_message,
                model_answer,
                safety_blocked,
            ]
        )

# =========[ 5) Flask API ]=========

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    user_role = data.get("user_role", "TECHNICIAN")
    machine_state = data.get("machine_state", "Healthy")
    vibration = float(data.get("vibration", 0.0))
    temperature = float(data.get("temperature", 0.0))
    current = float(data.get("current", 0.0))
    user_message = data.get("question", "")

    session_id = data.get("session_id") or str(uuid.uuid4())

    telemetry = build_telemetry(
        machine_state,
        vibration=vibration,
        temperature=temperature,
        current=current,
    )

    answer, contexts, safety_msg, blocked = generate_answer(
        user_message, user_role, telemetry
    )

    full_answer = safety_msg + "\n" + answer if safety_msg else answer

    log_interaction_to_csv(
        session_id=session_id,
        user_role=user_role,
        telemetry=telemetry,
        user_message=user_message,
        model_answer=full_answer,
        safety_blocked=blocked,
    )

    return jsonify(
        {
            "session_id": session_id,
            "answer": full_answer,
            "contexts": contexts,
            "safety_message": safety_msg,
            "safety_blocked": blocked,
        }
    )

if __name__ == "__main__":
    # للتجربة محلياً
    app.run(host="0.0.0.0", port=8000, debug=True)
