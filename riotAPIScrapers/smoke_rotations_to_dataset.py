#!/usr/bin/env python3
import os, sys, json, requests, datetime, pathlib
import pandas as pd

API_KEY  = os.getenv("RIOT_API_KEY") or sys.exit("Set RIOT_API_KEY")
PLATFORM = os.getenv("RIOT_PLATFORM", "na1")  # na1, euw1, kr, ...

# --- fetch (1 request) ---
url = f"https://{PLATFORM}.api.riotgames.com/lol/platform/v3/champion-rotations"
r = requests.get(url, headers={"X-Riot-Token": API_KEY}, timeout=10)
r.raise_for_status()
data = r.json()

# --- make dataset (no loops) ---
free = pd.Series(data.get("freeChampionIds", []), name="championId", dtype="int64")
new  = pd.Series(data.get("freeChampionIdsForNewPlayers", []), name="championId", dtype="int64")

df = pd.DataFrame({"championId": pd.unique(pd.concat([free, new], ignore_index=True))})
df["is_free"] = df["championId"].isin(free)
df["is_new_player_free"] = df["championId"].isin(new)
df["maxNewPlayerLevel"] = data.get("maxNewPlayerLevel")
df["platform"] = PLATFORM
df["fetched_at_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds")

# --- save ---
out_dir = pathlib.Path(os.getenv("ROTATIONS_OUT", os.path.expanduser("~/riot_out")))
out_dir.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

csv_path  = out_dir / f"champion_rotations_{PLATFORM}_{stamp}.csv"
json_path = out_dir / f"champion_rotations_{PLATFORM}_{stamp}.json"

df.to_csv(csv_path, index=False)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"platform": PLATFORM, **data}, f, ensure_ascii=False, indent=2)

print("Saved CSV:", csv_path)
print("Saved JSON:", json_path)
print(df.head(10).to_string(index=False))
