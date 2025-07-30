import json
import google.generativeai as genai
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()
# ─── Hard‑coded Gemini API key ─────────────────────────────────────────────────

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")

def generate_batch_answer(
    contexts: list[list[Document]],
    questions: list[str],
) -> list[str]:
    """
    Sends all questions+contexts in one shot, asks Gemini to reply
    with {"answers": [...]} JSON, then parses it robustly.
    """
    # 1) Build the prompt
    prompt = "You are a helpful assistant.\n\n"
    for idx, (q, ctx_docs) in enumerate(zip(questions, contexts), 1):
        ctx_text = "\n".join(d.page_content for d in ctx_docs)
        prompt += (
            f"Question {idx}:\n{q}\n\n"
            f"Context {idx}:\n{ctx_text}\n\n"
        )
    prompt += (
        "You are a specialist in insurance policy language. I will give you a list of questions and their raw answers. For each question, produce:1. a concise, precise “refined_answer” that uses exact numbers, terms, and conditions;  2. a “keywords” list of 3–5 short phrases capturing the core concepts;"
        "Please **only** return a JSON object with this schema:\n"
        '{"answers": ["Answer to Q1", "Answer to Q2", ...]}\n'
        "Do not include any additional text, explanation, or formatting."
    )

    # 2) Call Gemini
    response = model.generate_content(prompt)
    raw = response.text.strip()

    # 3) Sanitize: if it’s not valid JSON, try to pull out the first {...} block
    json_str = raw
    if not raw.startswith("{"):
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = raw[start : end + 1]

    # 4) Parse
    try:
        payload = json.loads(json_str)
        answers = payload.get("answers")
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise ValueError(f"Wrong format: {payload!r}")
        return [str(a).strip() for a in answers]

    except Exception as e:
        print("Gemini batch error:", e, "| raw response:", raw[:200])
        # 5) Fallback: split by markers in the raw text
        fallback = []
        for i in range(1, len(questions) + 1):
            marker = f"Answer {i}:"
            parts  = raw.split(marker, 1)
            if len(parts) == 2:
                # take everything after marker up to next marker
                rest = parts[1]
                next_marker = f"Answer {i+1}:"
                text = rest.split(next_marker, 1)[0].strip()
                fallback.append(text)
            else:
                fallback.append("❌ Error")
        return fallback
