# cs171\_league\_analysts\_in\_training

## Members: Brandon Lee and John Paul Silvas

## Project Name: League Analysts in Training

## Description of Topic

### Topic: League of Legends champion strength predictions

### Question: Can we predict how strong champions are based off Patch Data?

League of Legends is a popular MOBA (multiplayer online battle arena) game that has existed since 2010, and still pretty popular. The game is usually rebalanced every 2-3 weeks with a list of buffs, nerfs, adjustments, bug fixes, and upcoming skin lines. This allows overpowered champions to be made more equal in strength, while allowing undertuned champions to become more playable. For this topic, we want to connect patch data with 2 tierlist rankings: Clustered Ranking of S-D, and a discrete numerical ranking (1-171) for each character in a patch. We will connect the win rate (%), pick rate(%), and ban rate(%) of each character in a patch to try and create a prediction of which characters are strong and weak.

## Data Collection

Public League of Legends Statistics Websites, Riot Games Developer API

John Paul: Grab patch data from LOL match data aggregating data

Brandon: Go into Riot API and grab champion statistics (JSON)

## Model Plans

John Paul: KNN Model for Champion Ranks

Brandon: K-Means Clustering for Champion Ranks

## Table of Contents

### Data Preprocessing

Our notebooks that we used to gather data and turn them into csv files for training can be found in the Data Preparation folder:
1. lol_patch_winrates.py
2. TierlistScraper.ipynb

### Model Construction

Our notebooks where we created and tested our machine learning models can be found in the Models folder:
1. KNNTierlist.ipynb
2. LoL_KMeans_TierList.ipynb

### Analysis and Visualization

Our notebooks where we analyzed and visualized the results of the KNN and KMeans models can be found in the Models folder:
1. ModelAnalysis.ipynb

## Installation Guide

# Cloning The Github

git clone https://github.com/<your-org\>/cs171_league_analysts_in_training.git
cd cs171_league_analysts_in_training

# Package Notes

All packages in this program are contained in the CS171 kernel. There is no need to install any other package to the environment.

# Notes

requirements.txt has all the python packages listed needed to run all of the scripts in the github folder.

To run lol_patch_winrates.ipynb (run .py if it doesn't work), you need a Riot Games Account to access the API.

Here is the link to the Riot Games Developer API website: https://developer.riotgames.com/

For lol_patch_winrates.py (data scraper), make sure to run this line in your terminal to ensure RIOT_API_KEY works (key expires 24 hours after issued, which is why we decided to use a Python file instead of a Jupyter Notebook)

# macOS/Linux
export RIOT_API_KEY=RGAPI-xxxxxxxx

# Windows (PowerShell)
setx RIOT_API_KEY RGAPI-xxxxxxxx

## How to run Scripts/Notebooks

# ModelAnalysis.ipynb (in Models)
This report relies on graphs, csvs, and other images distributed out the repo, and therefore should not be moved.

All of these are connected to the notebook through relative paths, and nothing will be properly displayed unless
every file is in original state in the directory system.

# KNNTierlist.ipynb (in Models)
1. This notebook needs to access the lolpatch_{patch}.csv files that can be found in Datasets/lol_patch_tierlist.
2. It trains off of these files for the KNN model.
3. It should then access the champion_winrate*.csv on each folder in Datasets/riot_outs.
4. It will then create predictions of each patch as knnrank*.csv in Datasets/lol_patch_tierlist.

# Lol_KMeans_TierList.ipynb (in Models)

1. Open the notebook and run the Setup and Environment cells.
2. In Paths (Launcher) cell, point CSV_COMBINED to your combined CSV if needed.
3. Use the Diagonstics Launcher to explore K (elbow/silhouette).
4. Use the Single-Patch Launcher (PATCH = "15.20", K=5, LOGIT=TRUE) to produce a tier list for one patch
5. Use the All-Patches Launcher to generate tier lists for every patch. 

# lol_patch_winrates.py (in DataPreparation)

for output for this file, we are going to cs171_league_analysts_in_training/Datasets/riot_out

If you want a different folder, set OUT_DIR with an .env file at repo root or make a one-off environment variable

# TierlistScraper.ipynb (in DataPreparation)

For this file, you will need downloaded html files to scrape from. 

These html files should be in the same folder as the program, and once the cells are run: it will output lolpatch_{patch}.csv files in the same folder.

# Optional (QuickFlight)
python3 - <<'PY'
import os, requests
plat = "na1"
url = f"https://{plat}.api.riotgames.com/lol/status/v4/platform-data"
r = requests.get(url, headers={"X-Riot-Token": os.getenv("RIOT_API_KEY","")}, timeout=10)
print(f"[{plat}] status:", r.status_code)
PY

output should be 200.

# Running data scraper

SMOKE=1 python3 riotAPIScrapers/lol_patch_winrates.py

smoke test to verify everything works, grabs a handful of matches and writes a small CSV to Datasets/riot_out/

Default Run: python3 riotAPIScrapers/lol_patch_winrates.py

Single Patch: python3 riotAPIScrapers/lol_patch_winrates.py --patch 15.20

Sampling (for more variables)

# Fewer players, fewer matches per player (faster)
python3 riotAPIScrapers/lol_patch_winrates.py --patch 15.20 --max-puuids 120 --matches-per-puuid 3

# Environment-style overrides (work with or without CLI)
export MAX_PUUIDS=300
export MATCHES_PER_PUUID=10
export MAX_MATCHES_PER_PATCH=400
python3 riotAPIScrapers/lol_patch_winrates.py --patch 15.20

Each row in each csv has the following columns: patch, championId, championName, games, wins, win_rate, pick_rate, ban_rate

After Scraping, run the All-Patches K-Means Tierlist notebook cell in LOL_KMeans_TierList.ipynb




