#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tierlist_all_patches.py
Build KMeans-based tier lists for EVERY patch in a combined CSV.

Inputs (one row per champion per patch):
  patch, championId, championName, games, wins, win_rate, pick_rate, ban_rate

Outputs:
  <OUT_DIR>/patch_<patch>/tierlist_patch_<patch>.csv
  <OUT_DIR>/patch_<patch>/tier_centers_patch_<patch>.csv
  <OUT_DIR>/patch_<patch>/kmeans_validation_<patch>.csv
  <OUT_DIR>/tierlist_all_patches.csv
  <OUT_DIR>/kmeans_validation_all_patches.csv

CLI:
  python3 tierlist_all_patches.py \
    --csv /path/to/champion_winrates_all_patches.csv \
    --k 5 --logit --out-dir /path/to/riot_out
"""

import argparse, os, re, csv
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.linear_model import LinearRegression


# ----------------------------- utils -----------------------------

def canon_patch(s: str) -> str | None:
    """Normalize '15.20' from any '15.20.x' / loose strings."""
    s = str(s).strip()
    m = re.search(r'(\d+)\.(\d+)', s)
    return f"{int(m.group(1))}.{int(m.group(2))}" if m else None

def numeric_patch_key(p: str) -> tuple[int, int]:
    return tuple(map(int, p.split(".")))

def logit_percent(arr, eps=1e-4):
    """Logit transform percent features (stabilize extremes)."""
    p = np.clip(np.asarray(arr, dtype=float) / 100.0, eps, 1 - eps)
    return np.log(p / (1 - p))

def prepare_features(df: pd.DataFrame, use_logit: bool) -> np.ndarray:
    cols = ["win_rate", "pick_rate", "ban_rate"]
    X_raw = (
        np.column_stack([logit_percent(df[c].values) for c in cols])
        if use_logit else
        df[cols].to_numpy(dtype=float)
    )
    return StandardScaler().fit_transform(X_raw)

def tier_letters(k: int) -> list[str]:
    base = ["S","A","B","C","D","E","F","G","H","I"]
    if k <= len(base): return base[:k]
    return base + [f"T{i}" for i in range(k - len(base))]


# ------------------------- diagnostics --------------------------

def kmeans_diagnostics(X: np.ndarray, labels: np.ndarray, df_with_tier: pd.DataFrame) -> dict:
    """Internal quality + construct validity for one patch."""
    uniq = np.unique(labels)
    if len(uniq) > 1:
        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
    else:
        sil = db = ch = np.nan

    # Monotonicity of mean WR across tiers (S > A > B ...). Higher tier listed first.
    g = (df_with_tier.groupby("tier")["win_rate"]
         .mean()
         .sort_values(ascending=False))
    mono = bool(all(g.values[i] >= g.values[i+1] for i in range(len(g)-1))) if len(g) > 1 else True

    # How much WR variance tiers explain (quick R^2)
    D = pd.get_dummies(df_with_tier["tier"], drop_first=True)
    y = pd.to_numeric(df_with_tier["win_rate"], errors="coerce").fillna(0).values
    Xr = D.values if D.shape[1] else np.zeros((len(df_with_tier), 1))
    r2 = LinearRegression().fit(Xr, y).score(Xr, y) if len(df_with_tier) > 1 else np.nan

    return {
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "tier_monotone": mono,
        "tier_wr_r2": float(r2),
    }

def stability_ari(X: np.ndarray, k: int, seeds=(1,11,21,31,41), sample_weight=None) -> float:
    """Mean Adjusted Rand Index across multiple random seeds."""
    label_sets = []
    for s in seeds:
        km = KMeans(n_clusters=k, n_init=20, random_state=s)
        km.fit(X, sample_weight=sample_weight)
        label_sets.append(km.labels_)
    aris = []
    for i in range(len(label_sets)):
        for j in range(i+1, len(label_sets)):
            aris.append(adjusted_rand_score(label_sets[i], label_sets[j]))
    return float(np.mean(aris)) if aris else np.nan


# --------------------------- core -------------------------------

def run_for_patch(df_patch: pd.DataFrame, patch: str, k: int, use_logit: bool,
                  weight_by_games: bool, out_dir: Path, random_state=42):
    """Cluster one patch, save CSVs, return (rows, centers, diag)."""
    if len(df_patch) < k:
        print(f"[skip] patch {patch}: rows={len(df_patch)} < k={k}")
        return None, None, None

    # Prepare features
    X = prepare_features(df_patch, use_logit=use_logit)

    # Sample weighting by sqrt(games)
    sample_weight = None
    if weight_by_games and "games" in df_patch.columns:
        sample_weight = np.sqrt(
            np.clip(pd.to_numeric(df_patch["games"], errors="coerce").fillna(1).values, 1, None)
        )

    # Fit KMeans
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    km.fit(X, sample_weight=sample_weight)
    labels = km.labels_

    # Compute cluster centers in ORIGINAL % space (weighted by games if enabled)
    centers = []
    for c in range(k):
        members = (labels == c)
        n = int(members.sum())
        if n == 0:
            wmr = wpr = wbr = 0.0
        else:
            if weight_by_games and "games" in df_patch.columns:
                ws = df_patch.loc[members, "games"].clip(lower=1).astype(float).values
                wmr = float(np.average(df_patch.loc[members, "win_rate"].values, weights=ws))
                wpr = float(np.average(df_patch.loc[members, "pick_rate"].values, weights=ws))
                wbr = float(np.average(df_patch.loc[members, "ban_rate"].values,  weights=ws))
            else:
                wmr = float(df_patch.loc[members, "win_rate"].mean())
                wpr = float(df_patch.loc[members, "pick_rate"].mean())
                wbr = float(df_patch.loc[members, "ban_rate"].mean())
        centers.append({
            "cluster": c, "n": n,
            "mean_win_rate": wmr, "mean_pick_rate": wpr, "mean_ban_rate": wbr
        })

    # Rank clusters by mean win_rate (then pick_rate) → assign S/A/B...
    centers_sorted = sorted(centers, key=lambda d: (-d["mean_win_rate"], -d["mean_pick_rate"]))
    letters = tier_letters(k)
    cluster_to_tier = {cinfo["cluster"]: (letters[i] if i < len(letters) else f"T{i}")
                       for i, cinfo in enumerate(centers_sorted)}

    # Per-row output
    rows_out = []
    dfp = df_patch.reset_index(drop=True)
    for i, row in dfp.iterrows():
        c = int(labels[i])
        rows_out.append({
            "patch": patch,
            "championId": int(row["championId"]) if str(row["championId"]).isdigit() else row["championId"],
            "championName": row.get("championName", ""),
            "games": int(pd.to_numeric(row.get("games", 0), errors="coerce")) if pd.notna(row.get("games", None)) else 0,
            "wins": int(pd.to_numeric(row.get("wins", 0), errors="coerce")) if pd.notna(row.get("wins", None)) else 0,
            "win_rate": float(pd.to_numeric(row.get("win_rate", 0.0), errors="coerce")),
            "pick_rate": float(pd.to_numeric(row.get("pick_rate", 0.0), errors="coerce")),
            "ban_rate": float(pd.to_numeric(row.get("ban_rate", 0.0), errors="coerce")),
            "cluster": c,
            "tier": cluster_to_tier.get(c, "U"),
        })

    # Diagnostics
    df_out = pd.DataFrame(rows_out)
    diag = kmeans_diagnostics(X, labels, df_out)
    diag["patch"] = patch
    diag["k"] = k
    diag["n_rows"] = int(len(df_out))
    diag["stability_ari"] = stability_ari(X, k, seeds=(1,11,21,31,41), sample_weight=sample_weight)

    # Save per-patch outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    per_patch = out_dir / f"tierlist_patch_{patch}.csv"
    with per_patch.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patch","championId","championName","games","wins",
            "win_rate","pick_rate","ban_rate","cluster","tier"
        ])
        w.writeheader()
        w.writerows(rows_out)

    centers_csv = out_dir / f"tier_centers_patch_{patch}.csv"
    with centers_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cluster","n","mean_win_rate","mean_pick_rate","mean_ban_rate"])
        w.writeheader()
        w.writerows(centers_sorted)

    diag_csv = out_dir / f"kmeans_validation_{patch}.csv"
    pd.DataFrame([diag]).to_csv(diag_csv, index=False)

    print(f"[saved] {per_patch}")
    print(f"[saved] {centers_csv}")
    print(f"[saved] {diag_csv}")

    return rows_out, centers_sorted, diag


# ---------------------------- main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="KMeans tierlist for every patch in the combined CSV")
    ap.add_argument("--csv", required=True, help="Combined CSV path (from scraper)")
    ap.add_argument("--k", type=int, default=5, help="Clusters per patch (default 5)")
    ap.add_argument("--logit", action="store_true", help="Apply logit transform to WR/PR/BR before scaling")
    ap.add_argument("--no-weight", action="store_true", help="Disable sqrt(games) sample weighting")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/riot_out"), help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # Load & clean data
    df = pd.read_csv(args.csv, dtype={"patch": str})
    print(f"[debug] loaded rows: {len(df)} from {args.csv}")
    df["patch"] = df["patch"].map(canon_patch)

    if "ban_rate" not in df.columns: df["ban_rate"] = 0.0
    if "games" not in df.columns: df["games"] = 1

    for c in ["win_rate","pick_rate","ban_rate","games","wins","championId"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["patch","win_rate","pick_rate","ban_rate"]).copy()
    patches = sorted(df["patch"].unique(), key=numeric_patch_key)
    print("[debug] patches after cleaning:", patches)

    all_rows, all_diags = [], []

    for p in patches:
        dfp = df[df["patch"] == p].copy()
        sub_out = out_dir / f"patch_{p}"
        rows, _, diag = run_for_patch(
            df_patch = dfp,
            patch = p,
            k = args.k,
            use_logit = args.logit,
            weight_by_games = (not args.no_weight),
            out_dir = sub_out
        )
        if rows:
            all_rows.extend(rows)
        if diag:
            all_diags.append(diag)

    # Save combined outputs
    if all_rows:
        combined_csv = out_dir / "tierlist_all_patches.csv"
        with combined_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "patch","championId","championName","games","wins",
                "win_rate","pick_rate","ban_rate","cluster","tier"
            ])
            w.writeheader()
            w.writerows(all_rows)
        print(f"[saved] {combined_csv}")

    if all_diags:
        diag_out = out_dir / "kmeans_validation_all_patches.csv"
        pd.DataFrame(all_diags).to_csv(diag_out, index=False)
        print(f"[saved] {diag_out}")

    if not all_rows:
        print("[warn] no tierlists produced — check K and data size.")


if __name__ == "__main__":
    main()
