"""
apply_model.py - score a flow file with a SAVED model and report alert rates.
Use it on data the model never touched: Monday (all benign -> pure false-positive
check) or the VM flow log (cross-dataset). If the file has attack labels it also
reports recall per label.

  python3 apply_model.py --model model_v3_lean_seen.joblib --flows monday_flows_5s.jsonl
"""
import argparse, json, collections
import numpy as np
import joblib
from train_model_v3 import features, NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--flows", required=True)
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    cols = [NAMES.index(n) for n in bundle["features"]]
    print("model features:", bundle["features"])

    t, lab, src, dst, X = [], [], [], [], []
    for line in open(a.flows):
        f = json.loads(line)
        if f.get("packet_count", 0) < 1:
            continue
        t.append(f["first_seen"]); lab.append(f.get("label", "NORMAL"))
        src.append(f["src"]); dst.append(f["dst"])
        X.append(features(f))
    t, lab, X = np.array(t), np.array(lab), np.array(X)
    hours = max((t.max() - t.min()) / 3600, 1e-9)
    print(f"{len(t):,} flows spanning {hours:.2f} h")

    Z = bundle["scaler"].transform(np.log1p(X[:, cols]))
    score = bundle["model"].decision_function(Z)

    for fpr, thr in sorted(bundle["thresholds"].items(), reverse=True):
        pred = score > thr
        print(f"\nthreshold for {fpr:.1%} target FPR (picked on training-period NORMAL): thr={thr:.3f}")
        normal = lab == "NORMAL"
        print(f"  NORMAL flows alerted: {int(pred[normal].sum()):,} of {int(normal.sum()):,} "
              f"= {pred[normal].mean():.3%}  ({pred[normal].sum()/hours:,.0f} false alerts / hour)")
        for L in np.unique(lab[~normal]):
            m = lab == L
            print(f"  recall on {L}: {pred[m].mean():.3f} of {int(m.sum())}")
        top = collections.Counter((src[i], dst[i]) for i in np.where(pred & normal)[0]).most_common(5)
        print("  top alerting pairs among NORMAL:", top)


if __name__ == "__main__":
    main()
