"""
quantize_eval.py - how much of baseline v1 survives FP8 (E4M3) and CIM-style MAC error?

Pipeline per flow (same as baseline v1):
    x -> log1p -> standardise            (DIGITAL, stays float)
      -> dot product  w . z              (the part meant for the CIM macro)
      -> + intercept -> score -> threshold   (DIGITAL)

Configurations
  float  : float weights, float inputs                      (baseline v1, reference)
  fp8w   : weights rounded to FP8 E4M3, inputs float
  fp8wx  : weights AND standardised inputs rounded to E4M3   (what the macro would see)
Noise sweep (on fp8wx): random error added inside the dot product, N trials per level.
  --noise-model product   : every product w_i*z_i gets an independent relative error
                            p_i * (1 + sigma * N(0,1))
  --noise-model fullscale : dot += sigma * FS * N(0,1), FS = 99.9th pct of |dot| on training data
Thresholds are re-picked on held-out NORMAL flows from the CLEAN version of each
configuration (1% and 0.1% target FPR) and then held FIXED during the noise sweep,
exactly like a deployed detector. Same seen-split rows as train_model_v3.py.

The noise model is an ASSUMPTION (Gaussian, independent). Replace sigma / shape with the
macro's measured error distribution when you have it (published figure: ~1.5%).
"""
import argparse, csv
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from train_model_v3 import load, make_split, NAMES


def q_e4m3(x):
    """Round to the nearest FP8 E4M3 value (1 sign, 4 exponent, 3 mantissa; bias 7;
    max 448; subnormals below 2^-6). Round-to-nearest-even, saturating."""
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    a = np.minimum(np.abs(x), 448.0)
    out = np.zeros_like(a)
    sub = a < 2.0 ** -6
    out[sub] = np.round(a[sub] / 2.0 ** -9) * 2.0 ** -9
    nz = ~sub
    e = np.floor(np.log2(a[nz]))
    step = 2.0 ** (e - 3)
    out[nz] = np.round(a[nz] / step) * step
    return s * np.minimum(out, 448.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", default="friday_flows_5s.jsonl")
    ap.add_argument("--model", default="baseline_v1/model_v3_lean_seen.joblib")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--noise-model", choices=["product", "fullscale"], default="product")
    ap.add_argument("--sigmas", default="0,0.5,1,1.5,2,3,5,7.5,10", help="percent, comma separated")
    ap.add_argument("--out", default="quant_results.csv")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    cols = [NAMES.index(n) for n in bundle["features"]]
    clf, sc = bundle["model"], bundle["scaler"]
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    print("features:", bundle["features"])

    # ---- rebuild exactly the same split as train_model_v3.py --split seen ----
    t, lab, src, X = load(a.flows, False)
    y = (lab != "NORMAL").astype(int)
    train, test = make_split("seen", t, lab)
    val_cut = np.quantile(t[train & (y == 0)], 0.8)
    fit = train & ((t <= val_cut) | (y == 1))
    val_neg = train & (t > val_cut) & (y == 0)
    yt = y[test]
    print(f"test flows {int(test.sum()):,} (attacks {int(yt.sum())}), val NORMAL {int(val_neg.sum()):,}")

    Z = sc.transform(np.log1p(X[:, cols]))
    assert np.allclose(Z @ w + b, clf.decision_function(Z)), "float score mismatch"

    wq = q_e4m3(w)
    Zq = q_e4m3(Z)
    print("\nWEIGHTS  float -> FP8 E4M3  (relative error)")
    for n, wf, wk in zip(bundle["features"], w, wq):
        print(f"  {n:16s} {wf:+.5f} -> {wk:+.5f}   {abs(wk-wf)/abs(wf):.2%}")
    rel = np.abs(Zq - Z) / np.maximum(np.abs(Z), 1e-12)
    print(f"input rounding error (median relative): {np.median(rel[test]):.2%};  |z| max {np.abs(Z).max():.1f}")

    fprs = {"1%": 0.01, "0.1%": 0.001}

    def thresholds(score):
        return {k: float(np.quantile(score[val_neg], 1 - f)) for k, f in fprs.items()}

    def metrics(score_test, thr):
        out = {"roc": roc_auc_score(yt, score_test), "pr": average_precision_score(yt, score_test)}
        for k, th in thr.items():
            pred = score_test > th
            out["rec" + k] = pred[yt == 1].mean()
            out["fpr" + k] = pred[yt == 0].mean()
        return out

    configs = {
        "float": Z @ w + b,
        "fp8w": Z @ wq + b,
        "fp8wx": Zq @ wq + b,
    }
    thr_by = {k: thresholds(v) for k, v in configs.items()}
    pred_float = {k: configs["float"][test] > th for k, th in thr_by["float"].items()}

    print("\nCLEAN (no noise)           ROC-AUC  PR-AUC | recall / FPR @1% | recall / FPR @0.1% | alert flips vs float @0.1%")
    rows = []
    for name, sc_all in configs.items():
        m = metrics(sc_all[test], thr_by[name])
        flips = int(((sc_all[test] > thr_by[name]["0.1%"]) != pred_float["0.1%"]).sum())
        print(f"  {name:22s} {m['roc']:.4f}  {m['pr']:.4f} |  {m['rec1%']:.3f} / {m['fpr1%']:.3%}  |  "
              f"{m['rec0.1%']:.3f} / {m['fpr0.1%']:.3%}  |  {flips}")
        rows.append({"config": name, "sigma_pct": 0, "trials": 1, **{k: v for k, v in m.items()}})

    # ---- noise sweep on the FP8 weights + FP8 inputs configuration ----
    P = (Zq * wq)[test]                       # the individual products w_i * z_i
    FS = float(np.percentile(np.abs((Zq * wq)[fit].sum(1)), 99.9))
    thr = thr_by["fp8wx"]
    rng = np.random.default_rng(0)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    print(f"\nNOISE SWEEP on fp8wx ({a.noise_model}, {a.trials} trials per level; mean ± std)")
    print("sigma%   ROC-AUC          PR-AUC           recall@1%       FPR@1%          recall@0.1%     FPR@0.1%")
    for sg in sigmas:
        s = sg / 100.0
        res = []
        for _ in range(a.trials if s > 0 else 1):
            if s == 0:
                dot = P.sum(1)
            elif a.noise_model == "product":
                dot = (P * (1.0 + s * rng.standard_normal(P.shape))).sum(1)
            else:
                dot = P.sum(1) + s * FS * rng.standard_normal(len(P))
            res.append(metrics(dot + b, thr))
        keys = ["roc", "pr", "rec1%", "fpr1%", "rec0.1%", "fpr0.1%"]
        mean = {k: np.mean([r[k] for r in res]) for k in keys}
        std = {k: np.std([r[k] for r in res]) for k in keys}
        print(f"{sg:5.1f}   " + "   ".join(
            f"{mean[k]:.4f}±{std[k]:.4f}" if k in ("roc", "pr") else f"{mean[k]:.3f}±{std[k]:.3f}" for k in keys))
        rows.append({"config": "fp8wx+noise", "sigma_pct": sg, "trials": len(res),
                     **{k: mean[k] for k in keys}, **{k + "_std": std[k] for k in keys}})

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("config", "sigma_pct", "trials"), k))
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader(); wr.writerows(rows)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
