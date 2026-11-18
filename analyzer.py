import json, math, os, re
import requests

POS = ["love","loved","great","good","perfect","thank","excited","reasonable","helps","clean","works","exactly","glad","quite good","sounds"]
NEG = ["not great","crashed","lost","frustrating","waste","too late","too expensive","not interested","cancel","fine.","nobody","sorry","messy","high","problem"]
BUY = ["pricing","quote","sign up","get started","send me the proposal","send me","go ahead","onboarding","discount","how long","before the end","today","integrate","demo"]
OBJ = ["too expensive","a little high","think about it","discuss it with","competitor","next month","not interested","cancel","maybe"]

def _count(text, words):
    return sum(text.count(w) for w in words)

def local_analysis(turns):
    cust = " ".join(t for s, t in turns if s == "customer").lower()
    pos, neg = _count(cust, POS), _count(cust, NEG)
    buy, obj = _count(cust, BUY), _count(cust, OBJ)
    questions = cust.count("?")
    compound = (pos - neg) / (pos + neg + 2)          # lexicon polarity in [-1, 1]
    # Logistic model: hand-calibrated weights over conversation features
    z = -0.5 + 0.2 * buy - 0.35 * obj + 1.5 * compound + 0.1 * questions
    prob = 1 / (1 + math.exp(-z))
    prob = min(0.95, max(0.03, prob))
    sentiment = "Positive" if compound > 0.25 else "Negative" if compound < -0.1 else "Neutral"
    return {"sentiment": sentiment, "sentiment_score": round(compound, 2),
            "purchase_probability": round(prob * 100),
            "features": {"positive_terms": pos, "negative_terms": neg, "buying_signals": buy,
                         "objections": obj, "customer_questions": questions}}

def gemini_analysis(call):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    transcript = "\n".join(f"{s.upper()}: {t}" for s, t in call["turns"])
    prompt = (f"You are a sales-call analyst. Product: {call['product']}.\nTranscript:\n{transcript}\n\n"
              'Return JSON only: {"sentiment":"Positive|Neutral|Negative","purchase_probability":0-100,'
              '"summary":str,"key_insights":[str],"customer_behavior":[str],"buying_signals":[str],'
              '"objections":[str],"next_action":str}')
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key}, timeout=40,
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"responseMimeType": "application/json"}})
    r.raise_for_status()
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(re.sub(r"^```json|```$", "", text.strip()))

def _fallback_text(call, loc):
    s, f = loc["sentiment"], loc["features"]
    return {
        "summary": f"{call['customer']} ({call['company']}) showed {s.lower()} sentiment with {f['buying_signals']} buying signals and {f['objections']} objections.",
        "key_insights": [f"{f['positive_terms']} positive vs {f['negative_terms']} negative expressions from the customer.",
                         f"Customer asked {f['customer_questions']} question(s), indicating {'high' if f['customer_questions']>1 else 'limited'} engagement."],
        "customer_behavior": ["Engaged and forward-looking." if s == "Positive" else "Cautious, comparing options." if s == "Neutral" else "Frustrated and disengaging."],
        "buying_signals": [], "objections": [],
        "next_action": {"Positive": "Send quote and onboarding plan immediately.", "Neutral": "Share comparison material and follow up in a week.",
                        "Negative": "Escalate the issue and attempt service recovery."}[s]}

def analyze(call):
    loc = local_analysis(call["turns"])
    result, engine = None, "Local NLP model"
    try:
        result = gemini_analysis(call)
    except Exception as e:
        print("Gemini unavailable, using local engine:", e)
    if result:
        engine = "Gemini + local ensemble"
        llm_p = float(result.get("purchase_probability", loc["purchase_probability"]))
        prob = round(0.6 * llm_p + 0.4 * loc["purchase_probability"])
        sentiment = result.get("sentiment", loc["sentiment"])
    else:
        result, prob, sentiment = _fallback_text(call, loc), loc["purchase_probability"], loc["sentiment"]
    return {**result, "sentiment": sentiment, "purchase_probability": prob,
            "sentiment_score": loc["sentiment_score"], "features": loc["features"], "engine": engine}
