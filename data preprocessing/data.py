import requests
import pandas as pd
import io
import time

# Define Coordinates for all 16 Major Locations in Malaysia
locations = {
    "Perlis (Kangar)":       {"lat": 6.4412, "lon": 100.1986},
    "Kedah (Alor Setar)":    {"lat": 6.1248, "lon": 100.3678},
    "Penang (George Town)":  {"lat": 5.4141, "lon": 100.3288},
    "Perak (Ipoh)":          {"lat": 4.5975, "lon": 101.0901},
    "Selangor (Shah Alam)":  {"lat": 3.0738, "lon": 101.5183},
    "Kuala Lumpur":          {"lat": 3.1319, "lon": 101.6841},
    "Putrajaya":             {"lat": 2.9264, "lon": 101.6964},
    "Negeri Sembilan (Seremban)": {"lat": 2.7258, "lon": 101.9424},
    "Melaka (Bandar Melaka)":{"lat": 2.1896, "lon": 102.2501},
    "Johor (Johor Bahru)":   {"lat": 1.4927, "lon": 103.7414},
    "Pahang (Kuantan)":      {"lat": 3.8126, "lon": 103.3256},
    "Terengganu (K. Terengganu)": {"lat": 5.3117, "lon": 103.1324},
    "Kelantan (Kota Bharu)": {"lat": 6.1254, "lon": 102.2381},
    "Sabah (Kota Kinabalu)": {"lat": 5.9804, "lon": 116.0735},
    "Sarawak (Kuching)":     {"lat": 1.5533, "lon": 110.3592}
}

all_data = []
print("--- 🚀 Starting HOURLY Data Collection for 'Optimizing Solar Yield' ---")

# Define the years you want to fetch (NASA limits hourly data to 1 year per request)
years_to_fetch = range(2016, 2026)

# --- FETCH LOOP ---
for state, coords in locations.items():
    print(f"\n📡 Fetching data for: {state}")
    
    for year in years_to_fetch:
        print(f"   ⏳ Downloading year {year}...")
        
        # Format the start and end dates for the specific year
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        

        url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,CLOUD_AMT,T2M,TS,RH2M,PS,WS10M,WD10M,PW,ALLSKY_SRF_ALB,PRECTOTCORR",
            "community": "RE",
            "longitude": coords['lon'],
            "latitude": coords['lat'],
            "start": start_date,
            "end": end_date,
            "format": "CSV"
        }
        
        try:
            # We added timeout=60 so it won't hang forever!
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code != 200:
                print(f"   ⚠️ API Error {response.status_code} for {year}. Skipping.")
                continue

            # Skip metadata header dynamically to find the data table
            content = response.text.split('\n')
            header_row = 0
            for i, line in enumerate(content):
                if "YEAR,MO,DY,HR" in line:
                    header_row = i
                    break
            
            # If the server returned an empty file by accident, catch it
            if header_row == 0:
                print(f"   ⚠️ NASA returned empty data for {year}. Skipping.")
                continue
            
            df = pd.read_csv(io.StringIO(response.text), header=header_row)
            
            # Basic Cleaning & Feature Engineering
            df['State'] = state
            df['Latitude'] = coords['lat']
            df['Longitude'] = coords['lon']
            
            # CHANGED: Combine Year, Month, Day, and Hour into a single valid Datetime column
            df['Datetime'] = pd.to_datetime(
                df['YEAR'].astype(str) + '-' + 
                df['MO'].astype(str).str.zfill(2) + '-' + 
                df['DY'].astype(str).str.zfill(2) + ' ' + 
                df['HR'].astype(str).str.zfill(2) + ':00:00'
            )
            
            # Rename columns from NASA codes to ML-friendly names
            df = df.rename(columns={
                'ALLSKY_SFC_SW_DWN': 'GHI_W_m2',
                'ALLSKY_SFC_SW_DNI': 'DNI_W_m2',
                'ALLSKY_SFC_SW_DIFF': 'DHI_W_m2',
                'CLOUD_AMT': 'Cloud_Coverage_Pct',
                'T2M': 'Temp_Air_2m_C',
                'TS': 'Temp_Skin_C',
                'RH2M': 'Relative_Humidity_Pct',
                'PS': 'Surface_Pressure_kPa',
                'WS10M': 'Wind_Speed_10m_m_s',
                'WD10M': 'Wind_Direction_10m_deg',
                'PW': 'Precipitable_Water_cm',
                'PRECTOTCORR': 'Rainfall_mm'
            })
            
            # Select final columns in a logical order (Using 'Datetime' instead of 'Date')
            cols_to_keep = [
                'Datetime', 'State', 'Latitude', 'Longitude', 
                'GHI_W_m2', 'DNI_W_m2', 'DHI_W_m2', 
                'Cloud_Coverage_Pct', 'Temp_Air_2m_C', 'Temp_Skin_C', 
                'Relative_Humidity_Pct', 'Surface_Pressure_kPa', 
                'Wind_Speed_10m_m_s', 'Wind_Direction_10m_deg', 
                'Precipitable_Water_cm', 'Rainfall_mm'
            ]
            
            clean_df = df[cols_to_keep]
            all_data.append(clean_df)
            
            # Respect NASA's server limits (Slightly longer sleep to be safe with loop)
            time.sleep(1.5) 
            
        except Exception as e:
            print(f"   ❌ Error processing {state} ({year}): {e}")

# --- SAVE TO CSV ---
final_df = pd.concat(all_data, ignore_index=True)

# Sort chronologically just to be safe
final_df = final_df.sort_values(by=['State', 'Datetime'])

final_df.to_csv("malaysia_solar_hourly_analytics.csv", index=False)

print("\n✅ DONE! Saved 'malaysia_solar_hourly_analytics.csv'.")
print(f"Total Rows: {len(final_df)} (Should be ~1,400,000 for 10 years across 16 states)")
print("\n--- Sample Data (First 5 Rows) ---")
print(final_df.head())


