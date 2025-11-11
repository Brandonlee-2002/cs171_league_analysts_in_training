import os, time, re, math, requests, collections, itertools


API_KEY = os.getenv("RIOT_API_KEY")
if not API_KEY:
    raise SystemExit("Missing RIOT_API_KEY. Set it as an environment variable.")

# ----- Tunables -----
PATCH          = "15.21"                  # e.g., "15.21"
QUEUES         = [420]                    # ranked solo; use [420, 440] to include Flex
REGION_CLUSTERS= ["americas","europe","asia"]  # add "sea" if needed
PLATFORMS      = ["na1","euw1","eun1","kr","br1","la1","la2","oc1","tr1","ru","jp1"]  # add/trim as desired
SEED_TIERS     = [("RANKED_SOLO_5x5","DIAMOND","I"),
                  ("RANKED_SOLO_5x5","DIAMOND","II"),
                  ("RANKED_SOLO_5x5","EMERALD","I"),
                  ("RANKED_SOLO_5x5","PLATINUM","I")]  # widen/narrow as you like
PAGES_PER_TIER = 3                         # more pages => more seeds
MATCHES_PER_PUUID = 50                     # pull this many recent matches per seed
MIN_GAMES_FOR_WR = 20                      # show WR only above this sample size
SLEEP = 0.05                                # be gentle; retry on 429s too
# ---------------------

def riot_get(url, params=None, max_retries=3):
    for i in range(max_retries):
        r = requests.get(url, headers={"X-Riot-Token": API_KEY}, params=params or {}, timeout=30)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "2")))
            continue
        if r.status_code >= 500:
            time.sleep(1.0 * (i+1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()

def extract_patch(game_version_str):
    # Examples: "15.21.431.1234", "Version 15.21.1.1234"
    m = re.search(r"(\d+)\.(\d+)\.", game_version_str)
    return f"{m.group(1)}.{m.group(2)}" if m else None

def dd_versions():
    return riot_get("https://ddragon.leagueoflegends.com/api/versions.json")

def dd_champion_map(version=None, lang="en_US"):
    # Map numeric champion key -> (id, name)
    if version is None:
        version = dd_versions()[0]
    data = riot_get(f"https://ddragon.leagueoflegends.com/cdn/{version}/data/{lang}/champion.json")
    by_key = {}
    for champ_id, obj in data["data"].items():
        by_key[obj["key"]] = (champ_id, obj["name"])
    return by_key

def league_entries(platform, queue, tier, division, page):
    # league-exp-v4: entries by queue/tier/division
    url = f"https://{platform}.api.riotgames.com/lol/league-exp/v4/entries/{queue}/{tier}/{division}"
    return riot_get(url, params={"page": page})

def summoner_by_encrypted_id(platform, encrypted_summoner_id):
    url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/{encrypted_summoner_id}"
    return riot_get(url)

def match_ids_by_puuid(region_cluster, puuid, count=100, queues=None, start_time=None, end_time=None):
    url = f"https://{region_cluster}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"start": 0, "count": min(count, 100)}
    # Try server-side queue/time filters; if empty, we’ll retry without queue
    if queues:
        # match-v5 param name is "queue"
        params["queue"] = queues[0] if len(queues) == 1 else None
    if start_time: params["startTime"] = start_time
    if end_time:   params["endTime"]   = end_time
    mids = riot_get(url, params)
    if not mids and "queue" in params:
        # Workaround for occasional queue filter bugs: drop it and filter client-side
        del params["queue"]
        mids = riot_get(url, params)
    return mids

def fetch_match(region_cluster, match_id):
    url = f"https://{region_cluster}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return riot_get(url)

def is_queue_ok(match_info, queues):
    return (not queues) or match_info.get("queueId") in queues

def compute_stats():
    import collections, time, re

    champ_picks = collections.Counter()
    champ_wins  = collections.Counter()
    total_games = 0

    # Map numeric champion key -> (id, name) for safety
    champ_map = dd_champion_map()  # key(str) -> (id, name)

    seen_match_ids = set()

    # 1) Build seed list (collect PUUIDs directly if available)
    seed_puuids = []
    for platform in PLATFORMS:
        for queue, tier, division in SEED_TIERS:
            for page in range(1, PAGES_PER_TIER + 1):
                try:
                    entries = league_entries(platform, queue, tier, division, page) or []
                except Exception:
                    continue

                for e in entries:
                    # Prefer new field on LeagueEntryDTO
                    if e.get("puuid"):
                        seed_puuids.append((platform, e["puuid"]))
                        continue
                    # Fallback: old encryptedSummonerId path
                    sid = e.get("summonerId")
                    if sid:
                        try:
                            s = summoner_by_encrypted_id(platform, sid)
                            seed_puuids.append((platform, s["puuid"]))
                            time.sleep(SLEEP)
                        except Exception:
                            pass

    # De-dup
    seen = set()
    puuids = []
    for platform, p in seed_puuids:
        if p not in seen:
            seen.add(p)
            puuids.append((platform, p))

    # Platform -> region cluster map
    platform_to_region = {
        "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas", "oc1": "sea",
        "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
        "kr": "asia", "jp1": "asia"
    }

    # 2) Pull matches, filter by patch+queue, aggregate
    for platform, puuid in puuids:
        region = platform_to_region.get(platform, "americas")
        if region not in REGION_CLUSTERS:
            continue

        mids = match_ids_by_puuid(region, puuid, MATCHES_PER_PUUID, queues=QUEUES)
        for mid in mids:
            if mid in seen_match_ids:
                continue
            seen_match_ids.add(mid)

            try:
                m = fetch_match(region, mid)
            except Exception:
                continue

            info = m.get("info", {})
            # Client-side queue check (in case we dropped server-side filter)
            if not is_queue_ok(info, QUEUES):
                continue

            # Patch check (e.g., "15.21.x")
            patch = extract_patch(info.get("gameVersion", ""))
            if patch != PATCH:
                continue

            total_games += 1
            for part in info.get("participants", []):
                cname = part.get("championName")
                if not cname and part.get("championId") is not None:
                    cname = champ_map.get(str(part["championId"]), (None, None))[1] or str(part["championId"])

                champ_picks[cname] += 1
                if part.get("win"):
                    champ_wins[cname] += 1

            time.sleep(SLEEP)

    # 3) Compute WR & PR
    rows = []
    for champ, picks in champ_picks.items():
        if picks < 1:
            continue
        wr = champ_wins[champ] / picks
        pr = picks / float(total_games) if total_games else 0.0
        if picks >= MIN_GAMES_FOR_WR:
            rows.append({
                "champion": champ,
                "patch": PATCH,
                "games": picks,
                "wins": champ_wins[champ],
                "win_rate": round(100 * wr, 2),
                "pick_rate": round(100 * pr, 3),
            })

    rows.sort(key=lambda r: (-r["pick_rate"], r["champion"]))
    return {"patch": PATCH, "total_games": total_games, "stats": rows}


if __name__ == "__main__":
    stats = compute_stats()
    print(f'Patch {stats["patch"]}: {stats["total_games"]} games sampled')
    # for r in stats["stats"][:30]:
        # print(f'{r["champion"]:>15s}  WR {r["win_rate"]:5.1f}%  PR {r["pick_rate"]:5.2f}%  (n={r["games"]})')
