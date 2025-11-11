# file: smoke_rotations.py
import os, sys, requests

API_KEY  = os.getenv("RIOT_API_KEY") or sys.exit("Set RIOT_API_KEY")
PLATFORM = os.getenv("RIOT_PLATFORM", "na1")  # na1, euw1, kr, ...

url = f"https://{PLATFORM}.api.riotgames.com/lol/platform/v3/champion-rotations"
r = requests.get(url, headers={"X-Riot-Token": API_KEY}, timeout=10)
print("HTTP:", r.status_code)
r.raise_for_status()
data = r.json()

# no loops: just direct indexing / len()
print("freeChampionIds count:", len(data.get("freeChampionIds", [])))
print("newPlayerFree count:",  len(data.get("freeChampionIdsForNewPlayers", [])))
print("maxNewPlayerLevel:",    data.get("maxNewPlayerLevel"))
