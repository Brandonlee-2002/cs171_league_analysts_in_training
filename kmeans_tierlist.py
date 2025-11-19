
#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

TIERS_5 = ["S","A","B","C","D"]
TIERS_6 = ["S","A","B","C","D","E"]
TIERS_7 = ["S","A","B","C","D","E","F"]

def logit_percent(p, eps=1e-4):
    # p is in % (0..100). Convert to logit in R.
    p = np.clip(p / 100.0, eps, 1 - eps)
    return np.log(p / (1 - p))

def choose_tier_labels(k):
    if k == 5: return TIERS_5
    if k == 6: return TIERS_6
    if k == 7: return TIERS_7
    # fallback: T1..Tk (T1 = best)
    return [f"T{i}" for i in range(1, k+1)]

def rank_clusters_by_center(centers_raw_pct, weights=(0.6, 0.3, 0.1)):
    """
    Rank clusters by a composite score computed in **raw % space** (WR, PR, BR).
    centers_raw_pct: array shape (k, 3) for [wr%, pr%, br%]
    """
    w_wr, w_pr, w_br = weights
    scores = w_wr*centers_raw_pct[:,0] + w_pr*centers_raw_pct[:,1] + w_br*centers_raw_pct[:,2]
    order = np.argsort(-scores)  # descending
    rank_of_cluster = np.empty_like(order)
    rank_of_cluster[order] = np.arange(len(order))  # 0 = best
    return rank_of_cluster, scores

def cluster_one_patch(df_patch, k, use_logit=False, weight_by_games=True, random_state=42):
    """
    df_patch columns: championId, championName, win_rate, pick_rate, ban_rate, games
    Returns: per-row labels, tier letters, centers (raw %) and mapping.
    """
    feats = df_patch[["win_rate","pick_rate","ban_rate"]].to_numpy(dtype=float)
    # Keep a copy in % space for center back-transform
    feats_pct = feats.copy()

    # Optional transform then scale
    if use_logit:
        feats = np.column_stack([logit_percent(df_patch["win_rate"].values),
                                 logit_percent(df_patch["pick_rate"].values),
                                 logit_percent(df_patch["ban_rate"].values)])
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)

    # Sample weights (downweight low-sample champs)
    sample_weight = None
    if weight_by_games and "games" in df_patch.columns:
        # sqrt or log1p temper extremes; choose one:
        sample_weight = np.sqrt(np.clip(df_patch["games"].values, 1, None))
        # sample_weight = np.log1p(df_patch["games"].values)

    # KMeans
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    km.fit(X, sample_weight=sample_weight)
    labels = km.labels_

    # Compute cluster centers back in **raw %** units (for ranking)
    centers_in_feat_space = scaler.inverse_transform(km.cluster_centers_)
    if use_logit:
        # inverse-logit to %: sigmoid(x)*100
        sigmoid = lambda z: 1.0/(1.0+np.exp(-z))
        centers_raw_pct = sigmoid(centers_in_feat_space) * 100.0
    else:
        centers_raw_pct = centers_in_feat_space  # already roughly in % units

    # Rank clusters -> tiers
    rank_of_cluster, scores = rank_clusters_by_center(centers_raw_pct)
    tiers = choose_tier_labels(k)
    cluster_to_tier = {c: tiers[rank_of_cluster[c]] for c in range(k)}

    return labels, cluster_to_tier, centers_raw_pct, scores

def run_for_patch(df, patch, k, use_logit, weight_by_games, out_dir):
    dfp = df[df["patch"] == patch].copy()
    if dfp.empty:
        print(f"[skip] patch {patch}: no rows")
        return None, None

    labels, c2t, centers_raw_pct, scores = cluster_one_patch(
        dfp, k=k, use_logit=use_logit, weight_by_games=weight_by_games
    )
    dfp["cluster"] = labels
    dfp["tier"] = dfp["cluster"].map(c2t)

    # Save tier list
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"tierlist_patch_{patch}.csv"
    cols = ["patch","championId","championName","win_rate","pick_rate","ban_rate","games","cluster","tier"]
    dfp[cols].to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # Save cluster centers
    centers_df = pd.DataFrame(centers_raw_pct, columns=["center_wr_pct","center_pr_pct","center_br_pct"])
    centers_df["cluster"] = np.arange(len(centers_df))
    centers_df["score"] = scores
    centers_df["tier"] = centers_df["cluster"].map(c2t)
    centers_df["patch"] = patch
    centers_csv = out_dir / f"tier_centers_patch_{patch}.csv"
    centers_df[["patch","cluster","tier","center_wr_pct","center_pr_pct","center_br_pct","score"]].to_csv(centers_csv, index=False)
    print(f"[saved] {centers_csv}")

    return dfp, centers_df

def main():
    ap = argparse.ArgumentParser(description="K-means tier list from WR/PR/BR (per patch)")
    ap.add_argument("--csv", required=True, help="Input CSV with columns: patch, championId, championName, win_rate, pick_rate, ban_rate, games")
    ap.add_argument("--k", type=int, default=5, help="Number of tiers/clusters (default 5)")
    ap.add_argument("--patch", default=None, help="Specific patch (e.g., '15.22'). If omitted and --each not set, uses latest.")
    ap.add_argument("--each", action="store_true", help="Cluster each patch separately and save multiple tierlists")
    ap.add_argument("--logit", action="store_true", help="Use logit transform on rates before scaling (often better)")
    ap.add_argument("--no-weight", action="store_true", help="Disable games-based sample weighting")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/riot_out/tierlists"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    # minimal sanity
    for col in ["patch","win_rate","pick_rate","ban_rate","championId","championName"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")
    if "games" not in df.columns:
        df["games"] = 1

    # ensure numeric
    for c in ["win_rate","pick_rate","ban_rate","games"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["win_rate","pick_rate","ban_rate","games"])

    # pick patch(es)
    patches = sorted(df["patch"].unique(), key=lambda p: tuple(int(x) if x.isdigit() else 0 for x in p.split(".")))
    if args.each:
        all_rows, all_centers = [], []
        for p in patches:
            res = run_for_patch(df, p, args.k, args.logit, not args.no_weight, out_dir)
            if res[0] is not None:
                all_rows.append(res[0]); all_centers.append(res[1])
        if all_rows:
            pd.concat(all_rows).to_csv(out_dir / "tierlist_all_patches.csv", index=False)
            pd.concat(all_centers).to_csv(out_dir / "tier_centers_all_patches.csv", index=False)
            print(f"[saved] {out_dir/'tierlist_all_patches.csv'}")
            print(f"[saved] {out_dir/'tier_centers_all_patches.csv'}")
    else:
        target_patch = args.patch or patches[-1]
        run_for_patch(df, target_patch, args.k, args.logit, not args.no_weight, out_dir)

if __name__ == "__main__":
    main()
