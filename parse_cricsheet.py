import os
import urllib.request
import zipfile
import json
import glob
import pandas as pd
import numpy as np

# Configuration
ZIP_URL = "https://cricsheet.org/downloads/ipl_json.zip"
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
ZIP_PATH = os.path.join(DATA_DIR, "ipl_json.zip")
OUTPUT_CSV = os.path.join(DATA_DIR, "ipl_data.csv")

# Team name normalization map
TEAM_MAP = {
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
    'Kings XI Punjab': 'Punjab Kings',
    'Delhi Daredevils': 'Delhi Capitals',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Deccan Chargers': 'Sunrisers Hyderabad', # Rebranded slot
}

# Venue to city mapping for missing city values
VENUE_TO_CITY = {
    "M.Chinnaswamy Stadium": "Bangalore",
    "M Chinnaswamy Stadium": "Bangalore",
    "MA Chidambaram Stadium, Chepauk": "Chennai",
    "MA Chidambaram Stadium": "Chennai",
    "Wankhede Stadium": "Mumbai",
    "Eden Gardens": "Kolkata",
    "Feroz Shah Kotla": "Delhi",
    "Arun Jaitley Stadium": "Delhi",
    "Rajiv Gandhi International Stadium, Uppal": "Hyderabad",
    "Rajiv Gandhi International Stadium": "Hyderabad",
    "Punjab Cricket Association Stadium, Mohali": "Chandigarh",
    "Punjab Cricket Association Bindra Stadium": "Chandigarh",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Chandigarh",
    "Punjab Cricket Association IS Bindra Stadium": "Chandigarh",
    "Sawai Mansingh Stadium": "Jaipur",
    "Narendra Modi Stadium": "Ahmedabad",
    "Sardar Patel Stadium, Motera": "Ahmedabad",
    "Dr DY Patil Sports Academy": "Mumbai",
    "Brabourne Stadium": "Mumbai",
    "Maharashtra Cricket Association Stadium": "Pune",
    "Subrata Roy Sahara Stadium": "Pune",
    "Dubai International Cricket Stadium": "Dubai",
    "Sharjah Cricket Stadium": "Sharjah",
    "Sheikh Zayed Stadium": "Abu Dhabi",
    "SuperSport Park": "Centurion",
    "Kingsmead": "Durban",
    "St George's Park": "Port Elizabeth",
    "Newlands": "Cape Town",
    "De Beers Diamond Oval": "Kimberley",
    "Buffalo Park": "East London",
    "New Wanderers Stadium": "Johannesburg",
    "OUTsurance Oval": "Bloemfontein",
    "Barabati Stadium": "Cuttack",
    "Vidarbha Cricket Association Stadium, Jamtha": "Nagpur",
    "Himachal Pradesh Cricket Association Stadium": "Dharamsala",
    "JSCA International Stadium Complex": "Ranchi",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "Visakhapatnam",
    "Saurashtra Cricket Association Stadium": "Rajkot",
    "Holkar Cricket Stadium": "Indore",
    "Green Park": "Kanpur",
    "Shaheed Veer Narayan Singh International Stadium": "Raipur",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": "Lucknow",
    "Ekana Cricket Stadium": "Lucknow"
}

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading data from {ZIP_URL}...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print("Download complete.")
    else:
        print("Zip file already exists. Skipping download.")

def extract_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    # Check if we already have files extracted to avoid slow re-extraction
    existing_files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    if len(existing_files) > 100:
        print(f"Already extracted {len(existing_files)} files. Skipping extraction.")
        return

    print("Extracting match JSON files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        # Extract only json files to raw/
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.json'):
                # Extract directly to RAW_DIR without nested folder
                file_info.filename = os.path.basename(file_info.filename)
                zip_ref.extract(file_info, RAW_DIR)
    print("Extraction complete.")

def parse_all_matches():
    print("Parsing match files...")
    json_files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON files.")
    
    all_deliveries = []
    skipped_no_winner = 0
    skipped_no_2nd_innings = 0
    
    for i, filepath in enumerate(json_files):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{len(json_files)} matches...")
            
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
                
        info = data.get('info', {})
        
        # 1. Skip if no winner (e.g. tie, no result, abandoned)
        outcome = info.get('outcome', {})
        winner = outcome.get('winner')
        if not winner:
            skipped_no_winner += 1
            continue
            
        # 2. Get and clean teams
        teams = info.get('teams', [])
        if len(teams) < 2:
            continue
        teams = [TEAM_MAP.get(t, t) for t in teams]
        winner = TEAM_MAP.get(winner, winner)
        
        # Verify winner is one of the teams
        if winner not in teams:
            continue
            
        # 3. Get City and Venue
        venue = info.get('venue', 'Unknown')
        city = info.get('city')
        if not city:
            # Try to map venue to city
            for key, val in VENUE_TO_CITY.items():
                if key.lower() in venue.lower():
                    city = val
                    break
            if not city:
                city = venue.split(',')[0].strip() # Fallback to venue prefix
        
        # Clean city name
        if city == "Bengaluru":
            city = "Bangalore"
            
        # 4. Get Match Year
        dates = info.get('dates', [])
        year = None
        if dates:
            date_str = dates[0]
            if '-' in date_str:
                try:
                    year = int(date_str.split('-')[0])
                except ValueError:
                    pass
        if not year:
            year = 2008 # Fallback
            
        innings_list = data.get('innings', [])
        
        # We need first innings total to calculate target for second innings
        first_innings_total = 0
        first_innings_found = False
        second_innings_data = None
        
        for inn_idx, inn_data in enumerate(innings_list):
            # Cricsheet JSON v1.x structure can contain a "super over" innings or multiple segments
            # We want to extract the team batting and find innings 1 and 2
            # Let's check for standard innings
            # In Cricsheet, sometimes target is in metadata or we can calculate it from innings 1
            batting_team = inn_data.get('team')
            batting_team = TEAM_MAP.get(batting_team, batting_team)
            
            # Avoid super overs or other elements
            overs = inn_data.get('overs', [])
            if not overs:
                continue
                
            # If this is the first innings of the match
            if not first_innings_found:
                first_innings_found = True
                # Sum up all runs in 1st innings
                for over_data in overs:
                    for delivery in over_data.get('deliveries', []):
                        runs_data = delivery.get('runs', {})
                        first_innings_total += runs_data.get('total', 0)
            else:
                # This must be the 2nd innings (chase)
                second_innings_data = inn_data
                break
                
        if not first_innings_found or not second_innings_data:
            skipped_no_2nd_innings += 1
            continue
            
        target = first_innings_total + 1
        
        # Process 2nd innings ball-by-ball
        chasing_team = TEAM_MAP.get(second_innings_data.get('team'), second_innings_data.get('team'))
        defending_team = teams[0] if teams[1] == chasing_team else teams[1]
        
        runs_scored_so_far = 0
        wickets_fallen = 0
        legal_balls_bowled = 0
        
        for over_data in second_innings_data.get('overs', []):
            over_num = over_data.get('over') # 0-indexed
            
            for delivery in over_data.get('deliveries', []):
                # Runs scored on this ball
                runs_data = delivery.get('runs', {})
                total_runs = runs_data.get('total', 0)
                runs_scored_so_far += total_runs
                
                # Wickets on this ball
                wickets = delivery.get('wickets', [])
                wickets_fallen += len(wickets)
                
                # Check legality of ball
                extras = delivery.get('extras', {})
                is_wide = 'wides' in extras
                is_noball = 'noballs' in extras
                is_legal = not (is_wide or is_noball)
                
                if is_legal:
                    legal_balls_bowled += 1
                
                # Over float representation (e.g. 14.3 means completed 14 overs, 3 balls of 15th over)
                completed_overs = legal_balls_bowled // 6
                balls_in_over = legal_balls_bowled % 6
                over_float = completed_overs + balls_in_over / 10
                
                # Match result label: 1 if chasing team won, 0 if lost
                result = 1 if winner == chasing_team else 0
                
                all_deliveries.append({
                    'match_id': os.path.splitext(os.path.basename(filepath))[0],
                    'year': year,
                    'batting_team': chasing_team,
                    'bowling_team': defending_team,
                    'venue': venue,
                    'city': city,
                    'over': over_float,
                    'runs_scored_so_far': runs_scored_so_far,
                    'wickets_fallen': wickets_fallen,
                    'target': target,
                    'result': result
                })
                
    print(f"Skipped matches: no winner = {skipped_no_winner}, no 2nd innings = {skipped_no_2nd_innings}")
    df = pd.DataFrame(all_deliveries)
    print(f"Extracted {len(df)} ball-by-ball records for 2nd innings.")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved parsed data to {OUTPUT_CSV}")

if __name__ == "__main__":
    download_data()
    extract_data()
    parse_all_matches()
