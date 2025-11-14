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
from datetime import datetime, UTC

# Progress Bar

import time  # make sure this is imported

def fmt_secs(s):
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def print_bar(prefix, i, n, start_time=None, width=30):
    """Simple in-place progress bar with optional elapsed/ETA."""
    n = max(int(n), 1)
    i = min(int(i), n)
    frac = i / n
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)

    tail = ""
    if start_time is not None:
        elapsed = time.perf_counter() - start_time
        tail += f" | elapsed {fmt_secs(elapsed)}"
        if i > 0:
            eta = elapsed * (n - i) / i
            tail += f" | eta {fmt_secs(eta)}"

    print(f"\r{prefix} [{bar}] {i}/{n}{tail}", end="", flush=True)
    if i >= n:
        print()  # newline


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

# ----- Patch selection (calendar → Riot season mapping) -----
REQUEST_MAJOR = int(os.getenv("REQUEST_MAJOR", "25"))  # you want 25.x
LOL_MAJOR = REQUEST_MAJOR - 10                         # -> 15.x

MINOR_START = int(os.getenv("MINOR_START", "1"))
MINOR_END   = int(os.getenv("MINOR_END", "20"))

PATCHES = [f"{LOL_MAJOR}.{i}" for i in range(MINOR_START, MINOR_END + 1)]

SMOKE = os.getenv("SMOKE") == "1"
if SMOKE:
    # one quick patch (change minor via SMOKE_MINOR if you want)
    SMOKE_MINOR = os.getenv("SMOKE_MINOR", str(MINOR_END))
    PATCHES = [f"{LOL_MAJOR}.{SMOKE_MINOR}"]

# Ranked Solo queue only. Add 440 to include Flex.
QUEUES = [420]

# Platform shards to sample seeds from (can expand later)
PLATFORMS = ["na1"]

# Tier/divisions to pull seeds from
SEED_TIERS = [
    # High Elo
    ("RANKED_SOLO_5x5", "CHALLENGER", ""),
    ("RANKED_SOLO_5x5", "GRANDMASTER", ""),
    ("RANKED_SOLO_5x5", "MASTER", ""),
    # Diamond Rank
    ("RANKED_SOLO_5x5", "DIAMOND",  "I"),
    ("RANKED_SOLO_5x5", "DIAMOND",  "II"),
    ("RANKED_SOLO_5x5", "DIAMOND",  "III"),
    ("RANKED_SOLO_5x5", "DIAMOND",  "IV"),
    # Emerald Rank
    ("RANKED_SOLO_5x5", "EMERALD",  "I"),
    ("RANKED_SOLO_5x5", "EMERALD",  "II"),
    ("RANKED_SOLO_5x5", "EMERALD",  "III"),
    ("RANKED_SOLO_5x5", "EMERALD",  "IV"),
    # Platinum Rank
    ("RANKED_SOLO_5x5", "PLATINUM", "I"),
    ("RANKED_SOLO_5x5", "PLATINUM", "II"),
    ("RANKED_SOLO_5x5", "PLATINUM", "III"),
    ("RANKED_SOLO_5x5", "PLATINUM", "IV"),
    # Gold Rank
    ("RANKED_SOLO_5x5", "GOLD", "I"),
    ("RANKED_SOLO_5x5", "GOLD", "II"),
    ("RANKED_SOLO_5x5", "GOLD", "III"),
    ("RANKED_SOLO_5x5", "GOLD", "IV"),
    # Silver Rank
    ("RANKED_SOLO_5x5", "SILVER", "I"),
    ("RANKED_SOLO_5x5", "SILVER", "II"),
    ("RANKED_SOLO_5x5", "SILVER", "III"),
    ("RANKED_SOLO_5x5", "SILVER", "IV"),
    # Bronze Rank
    ("RANKED_SOLO_5x5", "BRONZE", "I"),
    ("RANKED_SOLO_5x5", "BRONZE", "II"),
    ("RANKED_SOLO_5x5", "BRONZE", "III"),
    ("RANKED_SOLO_5x5", "BRONZE", "IV"),
    # Iron Rank
    ("RANKED_SOLO_5x5", "IRON", "I"),
    ("RANKED_SOLO_5x5", "IRON", "II"),
    ("RANKED_SOLO_5x5", "IRON", "III"),
    ("RANKED_SOLO_5x5", "IRON", "IV"),
    
]

# How many league pages to read per tier/div (1..N). Keep small at first.
PAGES_PER_TIER = 1

# How many recent matches to ask for per PUUID (count<=100). Keep small at first.
MATCHES_PER_PUUID = 5
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

# Cap sampling per champion & per patch (env-overridable)
TARGET_PER_CHAMP = int(os.getenv("TARGET_PER_CHAMP", "10"))       # 10 games per champ
MAX_MATCHES_PER_PATCH = int(os.getenv("MAX_MATCHES_PER_PATCH", "250"))  # safety stop
STOP_WHEN_ALL_CHAMPS_COMPLETE = os.getenv("STOP_WHEN_ALL_CHAMPS_COMPLETE", "0") == "1"
SAVE_COMBINED = os.getenv("SAVE_COMBINED", "1") == "1"



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

def collect_seed_puuids(max_puuids: int | None = None, shuffle: bool = True):
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

    # de-dup
    seen, puuids = set(), []
    for plat, p in seed:
        if p not in seen:
            seen.add(p); puuids.append((plat, p))

    if shuffle:
        import random
        random.shuffle(puuids)
    if max_puuids:
        puuids = puuids[:max_puuids]
    return puuids
    

def dd_versions():
    r = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=15)
    r.raise_for_status()
    return r.json()  # list, latest first

def dd_champion_map(version=None, lang="en_US"):
    """
    Returns mapping like: {"1": ("Annie","Annie"), ...} via Data Dragon.
    Key is numeric-as-string champion key; value is (championId, name).
    """
    if version is None:
        version = dd_versions()[0]
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/{lang}/champion.json"
    r = requests.get(url, timeout=15); r.raise_for_status()
    data = r.json()["data"]
    by_key = {}
    for champ_id, obj in data.items():
        by_key[obj["key"]] = (champ_id, obj["name"])
    return by_key

def dd_champion_ids():
    """Set of numeric champion IDs as ints, from Data Dragon."""
    by_key = dd_champion_map()
    return {int(k) for k in by_key.keys()}


from collections import Counter

from collections import Counter

def compute_patch_stats(target_patch: str, puuids,
                        target_per_champ: int = TARGET_PER_CHAMP,
                        max_matches_per_patch: int = MAX_MATCHES_PER_PATCH):
    champ_games = Counter()
    champ_wins  = Counter()
    champ_name  = {}

    remaining = {cid: target_per_champ for cid in dd_champion_ids()}

    seen_match_ids = set()
    total_matches = 0

    puuid_total = len(puuids)
    puuid_i = 0
    t_patch = time.perf_counter()
    last_report = 0

    for platform, puuid in puuids:
        puuid_i += 1
        # progress across players for this patch
        print_bar(f"PUUIDs [{target_patch}]", puuid_i, puuid_total, start_time=t_patch)

        region = PLATFORM_TO_REGION.get(platform, "americas")
        try:
            mids = match_ids_by_puuid(region, puuid, count=MATCHES_PER_PUUID, queues=QUEUES)
        except Exception:
            mids = []

        for mid in mids:
            if max_matches_per_patch and total_matches >= max_matches_per_patch:
                break

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

            patch = extract_patch(info.get("gameVersion", ""))
            if patch != target_patch:
                continue

            contributed = False
            for part in info.get("participants", []):
                cid = part.get("championId")
                if cid is None:
                    continue

                if cid not in remaining:
                    remaining[cid] = target_per_champ
                if remaining[cid] <= 0:
                    continue

                cname = part.get("championName") or str(cid)
                champ_name[cid] = cname
                champ_games[cid] += 1
                if part.get("win"):
                    champ_wins[cid] += 1

                remaining[cid] -= 1
                contributed = True

            if contributed:
                total_matches += 1
                if total_matches - last_report >= 25:
                    # light heartbeat so you can see it’s alive
                    print(f"\rMatches [{target_patch}] {total_matches}", end="", flush=True)
                    last_report = total_matches

            if STOP_WHEN_ALL_CHAMPS_COMPLETE and all(v <= 0 for v in remaining.values()):
                break
        else:
            continue
        break

    # tidy up line after the last heartbeat
    if last_report:
        print()

    rows = []
    for cid, games in champ_games.items():
        wins = champ_wins[cid]
        wr = 100.0 * wins / games
        pr = 100.0 * games / total_matches if total_matches else 0.0  # sample pick rate
        rows.append({
            "patch": target_patch,
            "championId": cid,
            "championName": champ_name.get(cid, str(cid)),
            "games": games,
            "wins": wins,
            "win_rate": round(wr, 2),
            "pick_rate": round(pr, 3),
        })

    rows.sort(key=lambda r: (-r["pick_rate"], r["championName"]))
    return {"patch": target_patch, "total_matches": total_matches, "stats": rows}

def save_csv_for_patch(stats, out_dir: Path, separate_dirs: bool = True, write_combined: bool = SAVE_COMBINED):
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    patch = stats["patch"]
    patch_dir = out_dir / f"patch_{patch}" if separate_dirs else out_dir
    patch_dir.mkdir(parents=True, exist_ok=True)

    per_patch = patch_dir / f"champion_winrates_{patch}_{stamp}.csv"
    header = ["patch","championId","championName","games","wins","win_rate","pick_rate"]

    # write per-patch file
    with per_patch.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in stats["stats"]:
            w.writerow(r)

    # write/append combined only if enabled
    if write_combined:
        all_csv = out_dir / "champion_winrates_all_patches.csv"
        write_header = not all_csv.exists()
        with all_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            for r in stats["stats"]:
                w.writerow(r)

    print(f"[saved] {per_patch}  (matches in patch={stats['total_matches']})")


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
    import sys, time, argparse

    # ---------- CLI overrides ----------
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--patch", help="Single, list, or range. E.g. '15.2' or '15.10-15.12' or '25.2' or '25.1,25.3'")
    ap.add_argument("--max-puuids", type=int, help="Cap seed players (default from env MAX_PUUIDS or 200)")
    ap.add_argument("--matches-per-puuid", type=int, help="Matches per player (1..100, overrides script default)")
    args, _ = ap.parse_known_args()

    # Map '25.x' (calendar-style) -> '15.x' (Riot season) automatically
    def _canon_patch(label: str) -> str:
        maj, mn = label.split(".")
        maj_i, mn_i = int(maj), int(mn)
        if maj_i >= 20:
            maj_i -= 10
        return f"{maj_i}.{mn_i}"

    # Support single, comma list, and minor range (within one major)
    if args.patch:
        sel = args.patch.strip()
        tmp = []
        for token in sel.split(","):
            token = token.strip()
            if "-" in token:
                a, b = token.split("-", 1)
                a = _canon_patch(a); b = _canon_patch(b)
                amj, amn = map(int, a.split("."))
                bmj, bmn = map(int, b.split("."))
                if amj != bmj:
                    print("Warning: patch range must stay within one major; using the left major.")
                start_m, end_m = sorted([amn, bmn])
                tmp.extend([f"{amj}.{i}" for i in range(start_m, end_m + 1)])
            else:
                tmp.append(_canon_patch(token))
        # de-dup while preserving order
        seen = set()
        PATCHES = [p for p in tmp if not (p in seen or seen.add(p))]

    # Seed/match caps (CLI > env > defaults)
    max_puuids = args.max_puuids if args.max_puuids is not None else int(os.getenv("MAX_PUUIDS", "200"))
    if args.matches_per_puuid is not None:
        # update the module-level default
        MATCHES_PER_PUUID = max(1, min(100, args.matches_per_puuid))

    # ---------- Run ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building seed players...")
    puuids = collect_seed_puuids(max_puuids=max_puuids)
    print(f"Seeds collected: {len(puuids)} (cap={max_puuids})")

    if not puuids:
        sys.exit("No seed PUUIDs found — widen PLATFORMS/SEED_TIERS/PAGES_PER_TIER or check your API key.")

    print("Patches to process:", PATCHES)
    if not PATCHES:
        sys.exit("PATCHES is empty — pass --patch or check config.")

    t_patches = time.perf_counter()
    total_patches = len(PATCHES)
    done = 0

    for patch in PATCHES:
        print(f"\n=== Patch {patch} ===")
        stats = compute_patch_stats(patch, puuids)
        if not stats or stats.get("total_matches", 0) == 0:
            print(f"[{patch}] No matches found with current sampling; skipping CSV.")
        else:
            save_csv_for_patch(stats, OUT_DIR)  # each patch -> its own folder/file

        done += 1
        try:
            print_bar("Patches", done, total_patches, start_time=t_patches)
        except TypeError:
            print_bar("Patches", done, total_patches)

    dt = time.perf_counter() - t_patches
    print(f"\nAll done in {fmt_secs(dt)}. Outputs in: {OUT_DIR}")
