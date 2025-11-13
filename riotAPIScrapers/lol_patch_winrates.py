#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Champion win rates & pick rates per patch (LoL Match-V5)
- Filters by info.gameVersion -> "<major>.<minor>"
- Outputs CSV per patch + an all-patches CSV

Safe defaults:
- Small # of seed players, few matches per player
- Sequential requests with backoff on 429/5xx
- Queue filter set to Ranked Solo (420)

Environment vars you might set:
  RIOT_API_KEY    : your RGAPI-... key (required)
  OUT_DIR         : where to save CSVs (default: ~/riot_out)
  SMOKE           : "1" to do a super-light test run
"""

import os, re, time, csv, json, math, datetime
from pathlib import Path
import requests
from collections import Counter, defaultdict
import sys, itertools

# Progress Bar

def print_bar(prefix, i, n, width=30):
    """Simple in-place progress bar for known totals."""
    n = max(n, 1)
    i = min(i, n)
    frac = i / n
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {i}/{n}")
    sys.stdout.flush()
    if i >= n:
        sys.stdout.write("\n")

def spinner(msg):
    """Spinner for unknown totals. Usage: spin = spinner('Msg'); next(spin) each tick; spinner_done() at end."""
    while True:
        for ch in "|/-\\":
            sys.stdout.write(f"\r{msg} {ch}")
            sys.stdout.flush()
            yield

def spinner_done(msg="done ✓"):
    sys.stdout.write(f"\r{msg}\n")
    sys.stdout.flush()


# ----------------------- Config -----------------------
API_KEY = os.getenv("RIOT_API_KEY")
if not API_KEY:
    raise SystemExit("Missing RIOT_API_KEY environment variable.")

OUT_DIR = Path(os.getenv("OUT_DIR", os.path.expanduser("~/riot_out")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Patch range: 25.1 to 25.20 (inclusive)
PATCHES = [f"25.{i}" for i in range(1, 21)]

# Ranked Solo queue only. Add 440 to include Flex.
QUEUES = [420]

# Platform shards to sample seeds from (can expand later)
PLATFORMS = ["na1", "euw1", "kr"]

# Tier/divisions to pull seeds from
SEED_TIERS = [
    ("RANKED_SOLO_5x5", "PLATINUM", "I"),
    ("RANKED_SOLO_5x5", "EMERALD",  "IV"),
    ("RANKED_SOLO_5x5", "DIAMOND",  "IV"),
]

# How many league pages to read per tier/div (1..N). Keep small at first.
PAGES_PER_TIER = 1

# How many recent matches to ask for per PUUID (count<=100). Keep small at first.
MATCHES_PER_PUUID = 10

# Minimum champion games in a patch to report WR (avoid noisy tiny samples)
MIN_GAMES_FOR_WR = 10

# polite delays & backoff
SLEEP = 0.05            # small pause between calls
MAX_RETRIES = 4

# “Smoke mode” makes a micro run (useful to verify everything works)
SMOKE = os.getenv("SMOKE") == "1"
if SMOKE:
    PATCHES = [PATCHES[-1]]    # just the latest requested (25.20)
    PLATFORMS = ["na1"]
    SEED_TIERS = [("RANKED_SOLO_5x5", "PLATINUM", "I")]
    PAGES_PER_TIER = 1
    MATCHES_PER_PUUID = 3
    MIN_GAMES_FOR_WR = 1
    print("[SMOKE] Running a minimal sample to verify the pipeline...\n")

# Map platform -> regional routing for match-v5
PLATFORM_TO_REGION = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "oc1": "sea",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
    "kr": "asia", "jp1": "asia"
}

# ------------------ Riot helpers ----------------------

def riot_get(url, params=None):
    """GET with Riot headers, 429 Retry-After handling, 5xx backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.get(url, headers={"X-Riot-Token": API_KEY}, params=params or {}, timeout=30)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "2"))
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(1.0 * attempt)
            continue
        r.raise_for_status()
        time.sleep(SLEEP)
        return r.json()
    # last attempt result if still failing
    r.raise_for_status()

def extract_patch(game_version: str):
    """
    Extract "major.minor" from strings like "25.7.456.1234" or "Version 25.7.1".
    Returns e.g. "25.7" or None if not found.
    """
    m = re.search(r"(\d+)\.(\d+)\.", game_version)
    return f"{m.group(1)}.{m.group(2)}" if m else None

def league_entries(platform, queue, tier, division, page):
    """
    League-v4 entries (by queue/tier/division) for seeds.
    Some payloads include 'puuid'. If not, use 'summonerId' fallback.
    """
    url = f"https://{platform}.api.riotgames.com/lol/league/v4/entries/{queue}/{tier}/{division}"
    return riot_get(url, params={"page": page})

def summoner_by_encrypted_id(platform, encrypted_summoner_id):
    """summoner-v4: resolve encryptedSummonerId -> {puuid, ...}"""
    url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/{encrypted_summoner_id}"
    return riot_get(url)

def match_ids_by_puuid(region_cluster, puuid, count=10, queues=None, start=None):
    """
    match-v5: list match ids for a PUUID.
    If queue filter returns empty (happens sometimes), retry without it and filter client-side.
    """
    url = f"https://{region_cluster}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"start": 0 if start is None else start, "count": min(max(count, 1), 100)}
    if queues and len(queues) == 1:
        params["queue"] = queues[0]

    ids = riot_get(url, params)
    if not ids and "queue" in params:
        # retry w/o queue filter
        del params["queue"]
        ids = riot_get(url, params)
    return ids

def fetch_match(region_cluster, match_id):
    url = f"https://{region_cluster}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return riot_get(url)

def is_queue_ok(info, queues):
    return (not queues) or info.get("queueId") in queues

# ------------------ Core logic ------------------------

def collect_seed_puuids():
    seed = []
    total_pages = len(PLATFORMS) * len(SEED_TIERS) * PAGES_PER_TIER
    done = 0

    for platform in PLATFORMS:
        for queue, tier, division in SEED_TIERS:
            for page in range(1, PAGES_PER_TIER + 1):
                try:
                    entries = league_entries(platform, queue, tier, division, page) or []
                except Exception:
                    entries = []
                done += 1
                print_bar("Seeding", done, total_pages)

                for e in entries:
                    if e.get("puuid"):
                        seed.append((platform, e["puuid"]))
                    elif e.get("summonerId"):
                        try:
                            s = summoner_by_encrypted_id(platform, e["summonerId"])
                            seed.append((platform, s["puuid"]))
                        except Exception:
                            pass

    # de-dup as before...
    seen, puuids = set(), []
    for plat, p in seed:
        if p not in seen:
            seen.add(p); puuids.append((plat, p))
    return puuids


def compute_patch_stats(target_patch: str, puuids):
    from collections import Counter

    champ_games, champ_wins, champ_name = Counter(), Counter(), {}
    seen_match_ids = set()
    total_matches = 0

    total_puuids = len(puuids)
    puuid_done = 0
    spin = spinner(f"Matches [{target_patch}]")  # start spinner

    for platform, puuid in puuids:
        region = PLATFORM_TO_REGION.get(platform, "americas")
        try:
            mids = match_ids_by_puuid(region, puuid, count=MATCHES_PER_PUUID, queues=QUEUES)
        except Exception:
            mids = []

        for mid in mids:
            if mid in seen_match_ids: 
                continue
            seen_match_ids.add(mid)

            try:
                match = fetch_match(region, mid)
            except Exception:
                continue

            info = match.get("info", {})
            if not is_queue_ok(info, QUEUES):
                continue
            if extract_patch(info.get("gameVersion", "")) != target_patch:
                continue

            total_matches += 1
            next(spin)  # tick spinner as we accept a match into this patch

            for p in info.get("participants", []):
                cid = p.get("championId")
                cname = p.get("championName") or str(cid)
                champ_name[cid] = cname
                champ_games[cid] += 1
                if p.get("win"):
                    champ_wins[cid] += 1

        puuid_done += 1
        print_bar(f"PUUIDs [{target_patch}]", puuid_done, total_puuids)

    spinner_done(f"Matches [{target_patch}] {total_matches} done ✓")

    # ... compute rows and return as you already do ...

    # Compose rows
    rows = []
    for cid, games in champ_games.items():
        wins = champ_wins[cid]
        wr = (wins / games) * 100.0
        # Pick rate = fraction of matches where the champion appeared
        pr = (games / total_matches) * 100.0 if total_matches else 0.0
        if games >= MIN_GAMES_FOR_WR:
            rows.append({
                "patch": target_patch,
                "championId": cid,
                "championName": champ_name.get(cid, str(cid)),
                "games": games,
                "wins": wins,
                "win_rate": round(wr, 2),
                "pick_rate": round(pr, 3),
            })

    # Sort by pick_rate desc then name
    # ... after you build `rows` and `total_matches` ...
    rows.sort(key=lambda r: (-r["pick_rate"], r["championName"]))
    return {"patch": target_patch, "total_matches": total_matches, "stats": rows}


def save_csv_for_patch(stats, out_dir: Path):
    """Write one CSV for a patch and also append to a combined CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)  # <— add this line
    patch = stats["patch"]
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    per_patch = out_dir / f"champion_winrates_patch_{patch}_{stamp}.csv"
    all_csv  = out_dir / "champion_winrates_all_patches.csv"

    header = ["patch","championId","championName","games","wins","win_rate","pick_rate"]

    # Per-patch file
    with per_patch.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in stats["stats"]:
            w.writerow(r)

    # Append to combined file (create header if missing)
    write_header = not all_csv.exists()
    with all_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        for r in stats["stats"]:
            w.writerow(r)

    print(f"[saved] {per_patch}  (matches in patch={stats['total_matches']})")
    return str(per_patch), str(all_csv)


def detect_major_from_live_matches(puuids):
    """Peek a few recent matches to learn the live 'major' (e.g., '15')."""
    for platform, puuid in puuids[:3]:  # just a few
        region = PLATFORM_TO_REGION.get(platform, "americas")
        try:
            ids = match_ids_by_puuid(region, puuid, count=1)
            if not ids:
                continue
            info = fetch_match(region, ids[0]).get("info", {})
            p = extract_patch(info.get("gameVersion",""))  # e.g. '15.20'
            if p and "." in p:
                return p.split(".")[0]  # '15'
        except Exception:
            pass
    return None

def sniff_patches(puuids, sample=3):
    seen = Counter()
    for platform, puuid in puuids[:3]:
        region = PLATFORM_TO_REGION.get(platform, "americas")
        try:
            ids = match_ids_by_puuid(region, puuid, count=sample)
            for mid in ids:
                info = fetch_match(region, mid).get("info", {})
                p = extract_patch(info.get("gameVersion",""))
                if p: seen[p] += 1
        except Exception:
            pass
    if seen:
        print("Patch sniff (top):", dict(seen.most_common(8)))




# ---------------------- Main --------------------------

if __name__ == "__main__":
    print("Building seed players...")
    puuids = collect_seed_puuids()
    if SMOKE:
        sniff_patches(puuids)
    total_patches = len(PATCHES)
    done = 0

    major = detect_major_from_live_matches(puuids) or "15"
    PATCHES = [f"{major}.{i}" for i in range(1, 21)]  # 15.1 .. 15.20 (or whatever major was detected)

if os.getenv("SMOKE") == "1":
    PATCHES = [f"{major}.20"]  # smoke = just the latest minor you want to test


    for patch in PATCHES:
        stats = compute_patch_stats(patch, puuids)
        if not stats or stats["total_matches"] == 0:
            print(f"[{patch}] No matches found with current sampling; skipping CSV.")
            continue
        save_csv_for_patch(stats, OUT_DIR)

        save_csv_for_patch(stats, OUT_DIR)
        done += 1
        print_bar("Patches", done, total_patches)

    print("All done.")

