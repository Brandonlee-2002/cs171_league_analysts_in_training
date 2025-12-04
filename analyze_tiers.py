#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_tiers.py

Analyze KMeans-based tier predictions:
- Per-patch tier summaries & boxplots
- Tier variance explained (R^2)
- Tier transition matrices across patches
- Next-patch generalization R^2 (map t+1 champions to t's cluster order)
- Top movers (tierΔ, winΔ, pickΔ)

Matplotlib only; one chart per figure; no explicit colors.
"""

import argparse, os, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- utils ----------------

def canon_patch(s: str) -> str | None:
    s = str(s).strip()
    m = re.search(r"(\d+)\.(\d+)", s)
    return f"{int(m.group(1))}.{int(m.group(2))}" if m else None

def patch_key(p: str):
    return tuple(map(int, p.split(".")))

def tier_order():
    return ["S","A","B","C","D","E","F","G","H","I"]

def tier_rank(t: str) -> int:
    # Higher rank = stronger; S highest
    order = tier_order()
    t = (t or "").upper()
    return len(order) - order.index(t) if t in order else 0

def load_tiers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patch": str})
    df["patch"] = df["patch"].map(canon_patch)
    for c in ["win_rate","pick_rate","ban_rate","games","wins"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["patch","championName","tier"]).copy()
    df["tier"] = df["tier"].astype(str).str.upper()
    df["tier_rank"] = df["tier"].map(tier_rank)
    return df

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patch": str})
    df["patch"] = df["patch"].map(canon_patch)
    for c in ["win_rate","pick_rate","ban_rate","games","wins"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["patch","championName"]).copy()

# ------------- per-patch summaries -------------

def per_patch_summary(df_tiers: pd.DataFrame, patch: str, out_dir: Path):
    sub = df_tiers[df_tiers["patch"] == patch].copy()
    if sub.empty: return False

    # Means/medians/counts
    g = (sub.groupby("tier")
            .agg(win_mean=("win_rate","mean"),
                 win_median=("win_rate","median"),
                 pr_mean=("pick_rate","mean"),
                 br_mean=("ban_rate","mean"),
                 n=("championName","count"))
            .reindex(tier_order()))
    out_csv = out_dir / f"tier_summary_{patch}.csv"
    g.dropna(how="all").to_csv(out_csv)

    # R^2 of tiers → win_rate (construct validity)
    D = pd.get_dummies(sub["tier"], drop_first=True)
    y = sub["win_rate"].values
    Xr = D.values if D.shape[1] else np.zeros((len(sub),1))
    r2 = LinearRegression().fit(Xr, y).score(Xr, y) if len(sub) > 1 else np.nan
    pd.DataFrame([{"patch":patch,"tier_winrate_R2":r2}]).to_csv(
        out_dir / f"tier_variance_R2_{patch}.csv", index=False
    )

    # Boxplot of WR by tier
    plt.figure()
    tiers = [t for t in tier_order() if t in sub["tier"].unique()]
    data = [sub.loc[sub["tier"]==t,"win_rate"].values for t in tiers]
    plt.boxplot(data, labels=tiers, showfliers=False)
    plt.xlabel("Tier")
    plt.ylabel("Win rate (%)")
    plt.title(f"Win-rate by Tier — Patch {patch}")
    plt.tight_layout()
    plt.savefig(out_dir / f"tier_winrate_box_{patch}.png", bbox_inches="tight")
    plt.close()
    return True

# --------- transitions & generalization ----------

def transition_matrix(df_tiers: pd.DataFrame, patch_a: str, patch_b: str, out_dir: Path):
    a = df_tiers[df_tiers["patch"] == patch_a][["championName","tier","tier_rank"]]
    b = df_tiers[df_tiers["patch"] == patch_b][["championName","tier","tier_rank"]]
    m = a.merge(b, on="championName", how="inner", suffixes=("_a","_b"))
    if m.empty: return

    # matrix tier_a -> tier_b
    tiers = tier_order()
    mat = pd.DataFrame(0, index=tiers, columns=tiers, dtype=int)
    for _, r in m.iterrows():
        ta, tb = r["tier_a"], r["tier_b"]
        if ta in tiers and tb in tiers:
            mat.loc[ta, tb] += 1
    mat.to_csv(out_dir / f"tier_transition_{patch_a}_to_{patch_b}.csv")

    # summary: stays / within +/-1 tier
    delta = m["tier_rank_b"] - m["tier_rank_a"]
    stay = int((delta == 0).sum())
    within1 = int((delta.abs() <= 1).sum())
    pd.DataFrame([{
        "patch_a": patch_a, "patch_b": patch_b,
        "n_common": int(len(m)),
        "stay_same": stay,
        "within_1_tier": within1,
        "stay_rate": round(stay / len(m), 3) if len(m) else np.nan,
        "within_1_rate": round(within1 / len(m), 3) if len(m) else np.nan
    }]).to_csv(out_dir / f"tier_transition_summary_{patch_a}_to_{patch_b}.csv", index=False)

def generalization_r2(df_tiers: pd.DataFrame, df_raw: pd.DataFrame, patch_a: str, patch_b: str, out_dir: Path):
    """
    Use patch A’s cluster ranking (by mean WR) to assign tiers in patch B,
    then compute R^2 (tiers -> WR) in patch B.
    """
    a = df_tiers[df_tiers["patch"] == patch_a][["championName","tier"]]
    # Rank A tiers by mean WR in A (ties -> pick mean)
    a_wr = df_raw[df_raw["patch"] == patch_a][["championName","win_rate","pick_rate"]]
    a_rank = a.merge(a_wr, on="championName", how="left").groupby("tier").agg(
        win_mean=("win_rate","mean"), pick_mean=("pick_rate","mean")
    ).sort_values(["win_mean","pick_mean"], ascending=False)
    letters = tier_order()
    a_tier_to_pos = {t:i for i,(t,_) in enumerate(zip(a_rank.index, range(len(a_rank))))}
    # Map B rows to A’s tier order by joining champ names
    b = df_raw[df_raw["patch"] == patch_b][["championName","win_rate"]].copy()
    b = b.merge(a, on="championName", how="left", suffixes=("","_a"))
    # If a champion didn’t exist in A data, drop for this eval
    b = b.dropna(subset=["tier"])
    # One-hot encode A-based tiers & compute R^2 on B’s WR
    D = pd.get_dummies(b["tier"], drop_first=True)
    y = b["win_rate"].values
    Xr = D.values if D.shape[1] else np.zeros((len(b),1))
    r2 = LinearRegression().fit(Xr, y).score(Xr, y) if len(b) > 1 else np.nan
    pd.DataFrame([{
        "patch_a": patch_a, "patch_b": patch_b,
        "n_eval": int(len(b)), "generalization_R2": r2
    }]).to_csv(out_dir / f"generalization_R2_{patch_a}_to_{patch_b}.csv", index=False)

def top_movers(df_tiers: pd.DataFrame, df_raw: pd.DataFrame, patch_a: str, patch_b: str, out_dir: Path, top_n=25):
    ta = df_tiers[df_tiers["patch"] == patch_a][["championName","tier_rank"]].rename(columns={"tier_rank":"rank_a"})
    tb = df_tiers[df_tiers["patch"] == patch_b][["championName","tier_rank"]].rename(columns={"tier_rank":"rank_b"})
    ra = df_raw[df_raw["patch"] == patch_a][["championName","win_rate","pick_rate"]].rename(columns={"win_rate":"wr_a","pick_rate":"pr_a"})
    rb = df_raw[df_raw["patch"] == patch_b][["championName","win_rate","pick_rate"]].rename(columns={"win_rate":"wr_b","pick_rate":"pr_b"})
    m = tb.merge(ta, on="championName", how="inner").merge(rb, on="championName").merge(ra, on="championName")
    m["tier_delta"] = m["rank_b"] - m["rank_a"]
    m["win_delta"]  = m["wr_b"]   - m["wr_a"]
    m["pick_delta"] = m["pr_b"]   - m["pr_a"]
    m = m.sort_values(["tier_delta","win_delta"], ascending=[False, False])
    m.head(top_n).to_csv(out_dir / f"top_movers_{patch_a}_to_{patch_b}.csv", index=False)

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="Analyze KMeans tiers across patches")
    ap.add_argument("--tiers-csv", required=True, help="tierlist_all_patches.csv")
    ap.add_argument("--raw-csv", required=True, help="champion_winrates_all_patches.csv")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/riot_out/analysis"), help="output dir")
    ap.add_argument("--patch-start"); ap.add_argument("--patch-end")
    ap.add_argument("--latest-k", type=int)
    ap.add_argument("--plots", action="store_true", help="emit per-patch WR-by-tier boxplots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df_tiers = load_tiers(Path(args.tiers_csv))
    df_raw   = load_raw(Path(args.raw_csv))

    patches = sorted(df_tiers["patch"].unique(), key=patch_key)
    if args.latest_k:
        patches = patches[-args.latest_k:]
    else:
        if args.patch_start:
            patches = [p for p in patches if patch_key(p) >= patch_key(canon_patch(args.patch_start))]
        if args.patch_end:
            patches = [p for p in patches if patch_key(p) <= patch_key(canon_patch(args.patch_end))]

    # Per-patch summaries (and optional plots)
    for p in patches:
        per_dir = out_dir / f"patch_{p}"
        per_dir.mkdir(parents=True, exist_ok=True)
        ok = per_patch_summary(df_tiers, p, per_dir)
        if args.plots and ok:
            # plot saved inside per_patch_summary
            pass

    # Pairwise transitions & generalization for consecutive patches
    for a, b in zip(patches[:-1], patches[1:]):
        per_dir = out_dir / f"patch_{a}_to_{b}"
        per_dir.mkdir(parents=True, exist_ok=True)
        transition_matrix(df_tiers, a, b, per_dir)
        generalization_r2(df_tiers, df_raw, a, b, per_dir)
        top_movers(df_tiers, df_raw, a, b, per_dir, top_n=25)

    print(f"Saved analyses under: {out_dir}")

if __name__ == "__main__":
    main()
