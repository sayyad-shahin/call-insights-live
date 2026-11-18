"""Turns a call recording into [["agent"|"customer", text], ...] using the Gemini API."""
import base64, json, os, re
import requests

PROMPT = ("Transcribe this sales call. The 'agent' is the company's sales or support rep; the 'customer' is the person being "
          "sold to or helped. Return JSON only: a list of [speaker, text] pairs in spoken order, speaker being exactly "
          "\"agent\" or \"customer\". Do not summarise or omit anything.")

def transcribe(path, mime):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY set. Add it to .env, or paste a transcript in the form.")
    audio = base64.b64encode(open(path, "rb").read()).decode()
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key}, timeout=180,
        json={"contents": [{"parts": [{"inline_data": {"mime_type": mime, "data": audio}}, {"text": PROMPT}]}],
              "generationConfig": {"responseMimeType": "application/json"}})
    if r.status_code != 200:
        raise RuntimeError(f"Gemini returned {r.status_code}: {r.text[:200]}")
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    turns = json.loads(re.sub(r"^```json|```$", "", text.strip()))
    clean = [["agent" if str(s).lower().startswith("a") else "customer", str(t).strip()] for s, t in turns if str(t).strip()]
    if not clean:
        raise RuntimeError("Transcription came back empty.")
    return clean
