"""
train_model_v2.py - honest baseline: train on the EARLY part of Friday, test on
the LATE part, so the model must generalise to an attack it has never seen.

  train period : start of capture -> gap between PortScan and DDoS
                 (NORMAL + PortScan; Bot dropped unless --include-bot)
  test period  : gap -> end of capture   (NORMAL + DDoS)

Labels: ATTACK = any non-NORMAL label from cic_replay.py (IP pair + time window).
Features (4, same as before): avg_size, size_variance, timing_variance, port_count.
  port_count = distinct destination ports <= 1024 (same rule as rules.py), so the
  attacker's random high source ports in reverse-direction flows do not count.
  log1p is applied to all four features, then StandardScaler (fit on train only).
  Both steps are digital pre-processing; only w.x runs on the CIM macro.
Threshold: chosen on held-out NORMAL flows from the end of the training period
  (target false-positive rates), never on the test set.
"""
import argparse, json
from datetime import datetime
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURES = ["avg_size", "size_variance", "timing_variance", "port_count"]
ATTACKER = "172.16.0.1"
MAX_PORT = 1024


def features(f):
    n = f["packet_count"]
    avg = f["total_bytes"] / n
    size_var = max(0.0, f["sum_of_squares"] / n - avg * avg)
    if n > 1:
        mg = f["sum_of_gaps"] / (n - 1)
        tim_var = max(0.0, f["sum_of_squared_gaps"] / (n - 1) - mg * mg)
    else:
        tim_var = 0.0
    pc = len({p for p in f["ports"] if p <= MAX_PORT})
    return [avg, size_var, tim_var, pc]


def load(path, include_bot):
    t, lab, src, X = [], [], [], []
    for line in open(path):
        f = json.loads(line)
        if f["packet_count"] < 1 or (f["label"] == "Bot" and not include_bot):
            continue
        t.append(f["first_seen"]); lab.append(f["label"]); src.append(f["src"])
        X.append(features(f))
    return np.array(t), np.array(lab), np.array(src), np.array(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="friday_flows_5s.jsonl")
    ap.add_argument("--include-bot", action="store_true")
    ap.add_argument("--out", default="model_v2")
    a = ap.parse_args()

    t, lab, src, X = load(a.inp, a.include_bot)
    y = (lab != "NORMAL").astype(int)
    print(f"loaded {len(y):,} flows: " + ", ".join(f"{l}={int((lab==l).sum()):,}" for l in np.unique(lab)))

    ps, dd = t[lab == "PortScan"], t[lab == "DDoS"]
    if len(ps) == 0 or len(dd) == 0 or ps.max() >= dd.min():
        raise SystemExit("PortScan must end before DDoS starts for this split; check labels")
    cutoff = (ps.max() + dd.min()) / 2
    print("split time:", datetime.fromtimestamp(cutoff))

    train = t <= cutoff
    test = ~train
    val_cut = np.quantile(t[train], 0.8)
    fit = train & ((t <= val_cut) | (y == 1))      # all positives stay in training
    val_neg = train & (t > val_cut) & (y == 0)     # held-out NORMAL for thresholds
    print(f"fit={fit.sum():,} (attacks {int(y[fit].sum())})  val NORMAL={val_neg.sum():,}  "
          f"test={test.sum():,} (attacks {int(y[test].sum())}, prevalence {y[test].mean():.4%})")

    # ---- what do the groups look like? (raw features, medians) ----
    print("\nGROUP MEDIANS (raw): n | avg_size size_var timing_var port_count")
    groups = {
        "NORMAL train": train & (y == 0), "NORMAL test": test & (y == 0),
        "PortScan": lab == "PortScan",
        "DDoS fwd (attacker->victim)": (lab == "DDoS") & (src == ATTACKER),
        "DDoS rev (victim->attacker)": (lab == "DDoS") & (src != ATTACKER),
    }
    if a.include_bot:
        groups["Bot"] = lab == "Bot"
    for name, m in groups.items():
        if m.sum():
            med = np.median(X[m], axis=0)
            print(f"  {name:28s} {int(m.sum()):7d} | " + "  ".join(f"{v:10.3f}" for v in med))

    Xt = np.log1p(X)

    def fit_eval(cols, verbose):
        sc = StandardScaler().fit(Xt[fit][:, cols])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(Xt[fit][:, cols]), y[fit])
        score = clf.decision_function(sc.transform(Xt[:, cols]))
        roc = roc_auc_score(y[test], score[test])
        ap_ = average_precision_score(y[test], score[test])
        names = [FEATURES[c] for c in cols]
        print(f"\n=== features: {names}")
        print(f"TEST  ROC-AUC {roc:.4f}   PR-AUC {ap_:.4f}   (PR-AUC of a random guesser = {y[test].mean():.4f})")
        print("weights (standardised space): " + ", ".join(f"{n} {w:+.3f}" for n, w in zip(names, clf.coef_[0]))
              + f", intercept {clf.intercept_[0]:+.3f}")
        res = {}
        for fpr in (0.01, 0.001):
            thr = np.quantile(score[val_neg], 1 - fpr)
            pred = score > thr
            tp = int((pred & test & (y == 1)).sum()); fn = int((~pred & test & (y == 1)).sum())
            fp = int((pred & test & (y == 0)).sum()); tn = int((~pred & test & (y == 0)).sum())
            rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1); real_fpr = fp / max(fp + tn, 1)
            print(f"  threshold for {fpr:.1%} FPR on val: recall {rec:.3f}  precision {prec:.3f}  "
                  f"TP {tp} FN {fn} FP {fp}  (test FPR {real_fpr:.3%})")
            for g in ("DDoS fwd (attacker->victim)", "DDoS rev (victim->attacker)"):
                m = groups[g] & test
                if m.sum():
                    print(f"      recall on {g}: {pred[m].mean():.3f} of {int(m.sum())}")
            res[fpr] = float(thr)
        return sc, clf, res

    sc, clf, thr = fit_eval([0, 1, 2, 3], True)
    fit_eval([0, 1, 2], False)   # ablation: is port_count doing all the work?

    joblib.dump({"model": clf, "scaler": sc, "features": FEATURES, "thresholds": thr,
                 "transform": "log1p"}, a.out + ".joblib")
    with open(a.out + "_weights.json", "w") as f:
        json.dump({"features": FEATURES, "transform": "log1p",
                   "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
                   "weights": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                   "thresholds_by_val_fpr": {str(k): v for k, v in thr.items()}}, f, indent=2)
    print(f"\nsaved {a.out}.joblib and {a.out}_weights.json (old model.joblib untouched)")


if __name__ == "__main__":
    main()
