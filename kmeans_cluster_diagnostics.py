#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kmeans_cluster_diagnostics.py
Outputs KMeans "loss" graphs for LoL champion data:
- Elbow curve (Sum of Square Errors) vs Number of clusters
- (Optional) Silhouette score vs K

Usage:
  python3 kmeans_cluster_diagnostics.py \
    --csv "/path/to/champion_winrates_all_patches.csv" \
    --patch 15.20 \
    --k-min 2 --k-max 10 --logit \
    --out-dir "/path/to/riot_out/plots"

  # All patches
  python3 kmeans_cluster_diagnostics.py --csv ... --each --k-min 2 --k-max 10

Notes:
- Features: win_rate, pick_rate, ban_rate (percent). Optional --logit transform.
- Always scales features with StandardScaler.
- Weights KMeans by sqrt(games) unless --no-weight is passed.
- Saves: elbow_sse_patch_<patch>.png  (and silhouette_patch_<patch>.png unless --no-sil is set)
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def logit_percent(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=float) / 100.0, eps, 1 - eps)
    return np.log(p / (1 - p))


def canon_patch(p):
    s = str(p).strip()
    m = re.search(r'(\d+)\.(\d+)', s)
    return f"{int(m.group(1))}.{int(m.group(2))}" if m else None


def numeric_patch_key(p):
    return tuple(map(int, p.split(".")))


def prepare_features(df, use_logit):
    feats = df[["win_rate", "pick_rate", "ban_rate"]].to_numpy(dtype=float)
    if use_logit:
        feats = np.column_stack([
            logit_percent(df["win_rate"].values),
            logit_percent(df["pick_rate"].values),
            logit_percent(df["ban_rate"].values),
        ])
    X = StandardScaler().fit_transform(feats)
    return X


def run_for_patch(df, patch, k_min, k_max, use_logit, weight_by_games, out_dir: Path, save_sil=True, random_state=42):
    dpp = df[df["patch"] == patch].copy()
    if dpp.empty:
        print(f"[skip] patch {patch}: no rows after cleaning")
        return None

    # Prepare features
    X = prepare_features(dpp, use_logit=use_logit)
    n = len(dpp)
    if n < 3:
        print(f"[skip] patch {patch}: too few rows ({n})")
        return None

    # sample weights
    sample_weight = None
    if weight_by_games and "games" in dpp.columns:
        sample_weight = np.sqrt(np.clip(pd.to_numeric(dpp["games"], errors="coerce").fillna(1).values, 1, None))

    # Respect k bounds
    k_max_eff = max(k_min, min(k_max, n - 1))

    ks, inertias, sils = [], [], []
    for k in range(k_min, k_max_eff + 1):
        try:
            km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
            km.fit(X, sample_weight=sample_weight)
            ks.append(k)
            inertias.append(km.inertia_)
            if save_sil and k >= 2 and k < n:
                sils.append(silhouette_score(X, km.labels_))
            elif save_sil:
                sils.append(np.nan)
            print(f"[{patch}] K={k:2d}  SSE={km.inertia_:,.1f}" + (f"  sil={sils[-1]: .4f}" if save_sil else ""))
        except Exception as e:
            print(f"[{patch}] K={k}: skipping ({e})")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Elbow plot: match the example labels
    plt.figure(figsize=(10, 4))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Sum of Square Errors")
    plt.title(f"Elbow Curve — Patch {patch}")
    elbow_png = out_dir / f"elbow_sse_patch_{patch}.png"
    plt.savefig(elbow_png, bbox_inches="tight")
    plt.close()
    print(f"[saved] {elbow_png}")

    # Optional silhouette
    if save_sil:
        valid = [(k, s) for k, s in zip(ks, sils) if not np.isnan(s)]
        if valid:
            kv, sv = zip(*valid)
            plt.figure(figsize=(10, 4))
            plt.plot(kv, sv, marker="o")
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette Score")
            plt.title(f"Silhouette vs K — Patch {patch}")
            sil_png = out_dir / f"silhouette_patch_{patch}.png"
            plt.savefig(sil_png, bbox_inches="tight")
            plt.close()
            print(f"[saved] {sil_png}")

    # CSV summary
    import csv
    diag_csv = out_dir / f"kmeans_diagnostics_patch_{patch}.csv"
    with diag_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patch", "K", "SSE", "silhouette"])
        for i, k in enumerate(ks):
            sse = inertias[i]
            sil = sils[i] if save_sil and i < len(sils) else ""
            w.writerow([patch, k, sse, sil])
    print(f"[saved] {diag_csv}")

    return {"patch": patch, "K": ks, "SSE": inertias, "silhouette": sils if save_sil else None}


def main():
    ap = argparse.ArgumentParser(description="KMeans elbow (SSE) and silhouette diagnostics")
    ap.add_argument("--csv", required=True, help="Combined CSV (patch,win_rate,pick_rate,ban_rate,games)")
    ap.add_argument("--patch", help="Patch to analyze (e.g., 15.20). If omitted and --each not set, use latest numerically.")
    ap.add_argument("--each", action="store_true", help="Run for every patch separately")
    ap.add_argument("--k-min", type=int, default=2, help="Minimum K")
    ap.add_argument("--k-max", type=int, default=10, help="Maximum K")
    ap.add_argument("--logit", action="store_true", help="Apply logit transform to WR/PR/BR before scaling")
    ap.add_argument("--no-weight", action="store_true", help="Disable sqrt(games) sample weighting")
    ap.add_argument("--no-sil", action="store_true", help="Do not compute/save silhouette plot")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/riot_out/plots"), help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # Read CSV as string to preserve "15.20"
    df = pd.read_csv(args.csv, dtype={"patch": str})
    print(f"[debug] loaded rows: {len(df)} from {args.csv}")

    # Normalize patch
    df["patch"] = df["patch"].map(canon_patch)

    # Ensure columns exist + numeric
    if "ban_rate" not in df.columns: df["ban_rate"] = 0.0
    if "games" not in df.columns: df["games"] = 1
    for c in ["win_rate", "pick_rate", "ban_rate", "games"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["patch","win_rate","pick_rate","ban_rate","games"]).copy()

    patches = sorted(df["patch"].unique(), key=numeric_patch_key)
    print("[debug] patches after cleaning:", patches)

    if args.each:
        for p in patches:
            sub = out_dir / f"patch_{p}"
            run_for_patch(df, p, args.k_min, args.k_max, args.logit, (not args.no_weight), sub, save_sil=(not args.no_sil))
    else:
        target = canon_patch(args.patch) if args.patch else patches[-1] if patches else None
        if not target:
            raise SystemExit("No patches after cleaning.")
        print("[debug] selecting patch:", target)
        run_for_patch(df, target, args.k_min, args.k_max, args.logit, (not args.no_weight), out_dir, save_sil=(not args.no_sil))


if __name__ == "__main__":
    main()
