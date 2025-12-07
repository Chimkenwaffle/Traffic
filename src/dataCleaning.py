import pandas as pd
import kagglehub
from datetime import datetime

# ALL DATA CLEANING AND ADDING IN HERE

def get_csv(path, csv):    
    path = kagglehub.dataset_download(path)
    df = pd.read_csv(path + csv)
    return df

def time_data(df):
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.loc[df["Start_Time"].notna()]
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
    df = df.loc[df["End_Time"].notna()]
    return df

def clean_num_data(df):
    num_cols = ["Severity", "Distance(mi)", "Temperature(F)", "Visibility(mi)", "Precipitation(in)"]
    for c in num_cols:
        if c in df.columns:
            df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Severity", "Distance(mi)", "Temperature(F)", "Visibility(mi)", "Precipitation(in)"])
    return df

def clean_data(df):
    df = df.dropna(subset=["County", "State", "Start_Lat", "Start_Lng"])
    df = time_data(df)
    df = clean_num_data(df)
    return df
    
def add_data(df):
    df.loc[:, "Hour"] = df["Start_Time"].dt.hour
    df.loc[:, "Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    df.loc[:, "Is_Night"] = ((df["Hour"] < 6) | (df["Hour"] >= 20)).astype(int)
    df.loc[:, "Day_of_Week"] = df["Start_Time"].dt.dayofweek
    df.loc[:, "Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
    return df

def normalize_Abbreviations(df):
    
    state_abbrev_to_full = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
    }
    
    if "State" in df.columns:
        sample_state = str(df["State"].iloc[0]) if len(df) > 0 else ""
        if len(sample_state) == 2:
            df["State"] = df["State"].map(state_abbrev_to_full).fillna(df["State"])
            print(f"Converted state abbreviations to full names")
        else:
            print(f"State names are already in full form")
    
    return df

def combine_drivers_data_and_population_change(df):
    
    print(f"Processing driver data with shape: {df.shape}")
    
    # Age groups for population counts
    age_group_columns = {
        '18 to 24 years': None,
        '25 to 34 years': None,
        '35 to 44 years': None,
        '45 to 64 years': None,
        '65 to 84 years': None,
        '85 to 99 years': None,
        '100 years and over': None
    }
    
    # Find the column index for each age group
    for i, col in enumerate(df.columns):
        if col in age_group_columns:
            age_group_columns[col] = i
            print(f"Found '{col}' at column index {i}")
    
    # The first row contains the actual column headers
    header_row = df.iloc[0].tolist()
    print(f"\nHeader row: {header_row[:10]}...")
    
    state_idx = header_row.index('State') if 'State' in header_row else None
    county_idx = header_row.index('County') if 'County' in header_row else None
    
    print(f"State column index: {state_idx}")
    print(f"County column index: {county_idx}")
    
    if state_idx is None or county_idx is None:
        print("ERROR: Could not find State or County in header row")
        return pd.DataFrame()
    
    # Skip first 2 rows and start with actual data
    df_data = df.iloc[2:].copy()
    df_data = df_data.reset_index(drop=True)
    
    orig_cols = df.columns.tolist()
    state_col = orig_cols[state_idx]
    county_col = orig_cols[county_idx]
    
    print(f"\nUsing column '{state_col}' for State")
    print(f"Using column '{county_col}' for County")
    
    df_data = df_data.rename(columns={
        state_col: 'State',
        county_col: 'County'
    })
    
    # Extract population columns for each age group
    renamed_age_cols = []
    for age_group, idx in age_group_columns.items():
        if idx is not None:
            col_name = orig_cols[idx]
            new_col_name = f'{age_group}_pop'
            df_data = df_data.rename(columns={col_name: new_col_name})
            df_data[new_col_name] = pd.to_numeric(df_data[new_col_name], errors='coerce')
            renamed_age_cols.append(new_col_name)
    
    # Calculate total population 16+
    df_data["People_16_plus"] = df_data[renamed_age_cols].sum(axis=1)
    
    # NOW find the "Change, 2010 to 2020" columns for population change
    change_cols_to_sum = []
    for age_group, age_col_idx in age_group_columns.items():
        if age_col_idx is not None:
            # Search for "Change, 2010 to 2020" in the next several columns
            for offset in range(0, 10):
                check_idx = age_col_idx + offset
                if check_idx < len(header_row):
                    if header_row[check_idx] == 'Change, 2010 to 2020':
                        change_col_name = orig_cols[check_idx]
                        change_cols_to_sum.append(change_col_name)
                        print(f"Found 'Change, 2010 to 2020' for {age_group} at column {check_idx}")
                        break
    
    # Calculate population change if we found the columns
    if len(change_cols_to_sum) > 0:
        print(f"\nProcessing {len(change_cols_to_sum)} population change columns")
        for col in change_cols_to_sum:
            df_data[col] = pd.to_numeric(
                df_data[col].astype(str).str.replace(',', '').str.strip(),
                errors='coerce'
            ).fillna(0)
        
        df_data["Decade_Population_Change"] = df_data[change_cols_to_sum].sum(axis=1)
    else:
        print("WARNING: Could not find population change columns")
        df_data["Decade_Population_Change"] = 0
    
    print(f"\nData sample after processing:")
    print(df_data[['State', 'County', 'People_16_plus']].head(10))
    
    # Group by State and County for population 16+
    drivers_county = (
        df_data.groupby(['State', 'County'], as_index=False)
        .agg({
            'People_16_plus': 'sum',
            'Decade_Population_Change': 'sum'
        })
        .rename(columns={
            'People_16_plus': 'Total_People_16_plus',
            'Decade_Population_Change': 'Total_Decade_Population_Change'
        })
    )
    
    # Clean and filter
    drivers_county = drivers_county.dropna(subset=["State", "County", "Total_People_16_plus"])
    drivers_county = drivers_county[drivers_county["Total_People_16_plus"] > 0]
    
    drivers_county["State"] = drivers_county["State"].astype(str).str.strip()
    drivers_county["County"] = drivers_county["County"].astype(str).str.strip()
    
    # Remove invalid entries
    drivers_county = drivers_county[
        (drivers_county["State"] != 'nan') & 
        (drivers_county["State"] != '') &
        (drivers_county["County"] != 'nan') & 
        (drivers_county["County"] != '')
    ]
    
    print(f"\nDriver data prepared: {drivers_county.shape}")
    print(f"Sample of processed data:")
    print(drivers_county.head(10))
    print(f"\nPopulation 16+ statistics:")
    print(drivers_county["Total_People_16_plus"].describe())
    
    return drivers_county

def clean_cars_data(df):
    df = df[df["Count"] > 0]
    df = df.dropna(subset=["Transaction County","Residential County", "Count"])
    return df

def do_driver_data(df):
    df = normalize_Abbreviations(df)
    df = combine_drivers_data_and_population_change(df)
    return df

def do_traffic_data(df):
    df = clean_data(df)
    df = add_data(df)
    df = normalize_Abbreviations(df)
    return df
