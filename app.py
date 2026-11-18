import io, json, os, re, uuid, datetime
import requests
import transcribe
from dotenv import load_dotenv
from flask import request, Flask, jsonify, send_from_directory, send_file, abort
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import analyzer

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder="static", static_url_path="")
DB = os.path.join(BASE, "data", "calls.json")
CALLS = {c["id"]: c for c in json.load(open(DB))}
app.config["MAX_CONTENT_LENGTH"] = 18 * 1024 * 1024
MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm"}

def save(): json.dump(list(CALLS.values()), open(DB, "w"), indent=1)

def parse_manual(text):
    turns = []
    for line in text.splitlines():
        m = re.match(r"\s*(agent|customer)\s*:\s*(.+)", line, re.I)
        if m: turns.append([m.group(1).lower(), m.group(2).strip()])
        elif line.strip() and turns: turns[-1][1] += " " + line.strip()
    if not turns: raise ValueError("Transcript must have lines starting with 'Agent:' or 'Customer:'.")
    return turns
CACHE = {}

def get_analysis(cid):
    if cid not in CALLS: abort(404)
    if cid not in CACHE: CACHE[cid] = analyzer.analyze(CALLS[cid])
    return CACHE[cid]

@app.get("/")
def index(): return app.send_static_file("index.html")

@app.get("/api/calls")
def calls():
    return jsonify([{k: v for k, v in c.items() if k != "turns"} for c in reversed(list(CALLS.values()))])

@app.get("/api/calls/<int:cid>")
def call(cid):
    if cid not in CALLS: abort(404)
    return jsonify(CALLS[cid])

@app.get("/api/calls/<int:cid>/analysis")
def analysis(cid): return jsonify(get_analysis(cid))

@app.get("/audio/<path:name>")
def audio(name): return send_from_directory(os.path.join(BASE, "audio"), name)

@app.get("/api/calls/<int:cid>/report")
def report(cid):
    c, a = CALLS.get(cid) or abort(404), get_analysis(cid)
    st = getSampleStyleSheet(); buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"Call Report #{cid}", leftMargin=48, rightMargin=48)
    el = [Paragraph(f"Call Analysis Report #{cid}", st["Title"])]
    meta = [["Customer", c["customer"]], ["Company", c["company"]], ["Agent", c["agent"]], ["Date", c["date"]],
            ["Product", c["product"]], ["Sentiment", f"{a['sentiment']} ({a['sentiment_score']})"],
            ["Purchase probability", f"{a['purchase_probability']}%"], ["Engine", a["engine"]]]
    t = Table(meta, colWidths=[130, 360])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.lightgrey), ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke)]))
    el += [t, Spacer(1, 12), Paragraph("Summary", st["Heading2"]), Paragraph(a["summary"], st["BodyText"])]
    for title, key in [("Key insights", "key_insights"), ("Customer behavior", "customer_behavior"),
                       ("Buying signals", "buying_signals"), ("Objections", "objections")]:
        if a.get(key):
            el.append(Paragraph(title, st["Heading2"]))
            el += [Paragraph("• " + x, st["BodyText"]) for x in a[key]]
    el += [Paragraph("Recommended next action", st["Heading2"]), Paragraph(a["next_action"], st["BodyText"]),
           Paragraph("Transcript", st["Heading2"])]
    el += [Paragraph(f"<b>{s.title()}:</b> {x}", st["BodyText"]) for s, x in c["turns"]]
    doc.build(el); buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"call_{cid}_report.pdf", mimetype="application/pdf")

@app.get("/api/status")
def status():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    if not key:
        if os.path.exists(os.path.join(BASE, ".env.txt")):
            hint = "Found '.env.txt'. Rename it to '.env' (Windows hides the .txt part), then restart."
        elif not os.path.exists(os.path.join(BASE, ".env")):
            hint = "No .env file next to app.py. Copy .env.example to .env, add your key, restart."
        else:
            hint = ".env found but GEMINI_API_KEY is empty. Paste the key after the = sign, save, restart."
        return jsonify(ok=False, hint="Gemini key missing. " + hint)
    try:
        r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models/{model}", headers={"x-goog-api-key": key}, timeout=8)
    except Exception:
        return jsonify(ok=None, hint="Key found, but Google could not be reached. Check your internet.")
    if r.status_code in (400, 401, 403):
        return jsonify(ok=False, hint="Google rejected the key. Create a new one at aistudio.google.com/apikey.")
    if r.status_code == 404:
        return jsonify(ok=False, hint=f"Model '{model}' not found. Set GEMINI_MODEL in .env to a current model.")
    return jsonify(ok=True, hint=f"Gemini connected ({model}).")

@app.errorhandler(413)
def too_big(e): return jsonify(error="File too large (max 18 MB)."), 413

@app.post("/api/upload")
def upload():
    f = request.files.get("audio")
    if not f or not f.filename: return jsonify(error="Choose an audio file."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in MIME: return jsonify(error="Use WAV, MP3, M4A, OGG, FLAC or WEBM."), 400
    name = uuid.uuid4().hex[:12] + ext
    path = os.path.join(BASE, "audio", name); f.save(path)
    try:
        manual = request.form.get("transcript", "").strip()
        turns = parse_manual(manual) if manual else transcribe.transcribe(path, MIME[ext])
    except Exception as e:
        os.remove(path); return jsonify(error=str(e)), 400
    cid = max(CALLS, default=0) + 1
    g = lambda k, d: request.form.get(k, "").strip() or d
    CALLS[cid] = {"id": cid, "customer": g("customer", "Unknown customer"), "company": g("company", "-"), "product": g("product", "Product"),
                  "agent": g("agent", "-"), "date": g("date", str(datetime.date.today())), "file": name, "turns": turns}
    save()
    return jsonify(id=cid)

@app.post("/api/calls/<int:cid>/swap")
def swap(cid):
    if cid not in CALLS: abort(404)
    CALLS[cid]["turns"] = [["customer" if s == "agent" else "agent", t] for s, t in CALLS[cid]["turns"]]
    CACHE.pop(cid, None); save()
    return jsonify(ok=True)

@app.delete("/api/calls/<int:cid>")
def delete(cid):
    c = CALLS.pop(cid, None) or abort(404)
    CACHE.pop(cid, None); save()
    try: os.remove(os.path.join(BASE, "audio", c["file"]))
    except OSError: pass
    return jsonify(ok=True)

if __name__ == "__main__":
    k = os.getenv("GEMINI_API_KEY", "").strip()
    print("Gemini key:", f"loaded (ends ...{k[-4:]})" if k else "NOT FOUND - see the status box in the portal")
    app.run(port=5000, debug=False)
