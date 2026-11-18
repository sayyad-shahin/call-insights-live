"""Measures the local model against labelled calls: python evaluate.py"""
import json
from analyzer import local_analysis
rows = json.load(open("data/eval_calls.json"))
labels = ["Positive", "Neutral", "Negative"]
conf = {a: {b: 0 for b in labels} for a in labels}
brier, hits, probs, outs = 0, 0, [], []
for true_s, bought, text in rows:
    r = local_analysis([("customer", text)])
    conf[true_s][r["sentiment"]] += 1
    p = r["purchase_probability"] / 100
    brier += (p - bought) ** 2; hits += (p >= 0.5) == bool(bought)
    probs.append(p); outs.append(bought)
    if r["sentiment"] != true_s or (p >= 0.5) != bool(bought):
        print(f"MISS  true={true_s}/{bought}  got={r['sentiment']}/{round(p*100)}%  | {text[:70]}")
n = len(rows)
print(f"\nSentiment accuracy : {sum(conf[l][l] for l in labels)}/{n}")
print("Confusion (rows=true, cols=predicted):", *[f"\n  {a:9}" + "".join(f"{conf[a][b]:4}" for b in labels) for a in labels])
print(f"\nPurchase accuracy @50%: {hits}/{n}")
print(f"Brier score: {brier/n:.3f}  (0 = perfect, 0.25 = coin flip)")
pos = [p for p, o in zip(probs, outs) if o]; neg = [p for p, o in zip(probs, outs) if not o]
auc = sum((a > b) + .5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
print(f"AUC (ranking quality): {auc:.2f}  (1 = perfect, 0.5 = random)")
