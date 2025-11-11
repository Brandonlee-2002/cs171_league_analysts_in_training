#!/usr/bin/env python3
import os, sys, requests

# --- Config ---
PLATFORM = os.getenv("RIOT_PLATFORM", "na1")   # na1, euw1, kr, etc.
MODE = os.getenv("RIOT_MODE", "rotations")     # "rotations" or "status"
API_KEY = os.getenv("RIOT_API_KEY")            # set this in your shell
# --------------

if not API_KEY:
    sys.exit("Missing RIOT_API_KEY environment variable.")

if MODE == "status":
    url = f"https://{PLATFORM}.api.riotgames.com/lol/status/v4/platform-data"
else:
    url = f"https://{PLATFORM}.api.riotgames.com/lol/platform/v3/champion-rotations"

try:
    r = requests.get(url, headers={"X-Riot-Token": API_KEY}, timeout=10)
    print("URL:", url)
    print("HTTP:", r.status_code)

    if r.status_code == 200:
        data = r.json()
        # Show a tiny summary depending on mode
        if MODE == "status":
            incidents = sum(len(x.get("incidents", [])) for x in data.get("services", []))
            print("Services:", len(data.get("services", [])), "| Incidents right now:", incidents)
        else:
            print("freeChampionIds:", len(data.get("freeChampionIds", [])),
                  "| newPlayerFree:", len(data.get("freeChampionIdsForNewPlayers", [])))

        # Peek at rate limit headers (may be absent for status)
        print("X-App-Rate-Limit:", r.headers.get("X-App-Rate-Limit"))
        print("X-App-Rate-Limit-Count:", r.headers.get("X-App-Rate-Limit-Count"))
        print("Retry-After:", r.headers.get("Retry-After"))
        sys.exit(0)

    elif r.status_code == 401:
        print("401 Unauthorized — key expired or missing.")
    elif r.status_code == 403:
        print("403 Forbidden — key invalid or not allowed for this endpoint.")
    elif r.status_code == 429:
        print("429 Rate limited — wait and retry. Retry-After:", r.headers.get("Retry-After"))
    else:
        print("Body (first 300 chars):", r.text[:300])

except requests.exceptions.RequestException as e:
    print("Request failed:", e)
    sys.exit(2)
