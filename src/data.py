import pandas as pd
import kagglehub

# ALL DATA CLEANING AND ADDING IN HERE

# gets the data from kaggle using the path and csv
def get_data(path, csv):    
    # gets the data set from kagglehub
    path = kagglehub.dataset_download(path)
    # stores it into dataframe called df
    df = pd.read_csv(path + csv)
    
    return df

# cleans "Start_Time" and "End_Time"
def time_data(df):
    # attempts to convert data into date time format deletes if unable to
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time"])

    # Probably not needed ?? VVVVV
    # b/c we alr convert start time and just use that?? either way cleans out any unreadable end times

    # attempts to convert data into date time format deletes if unable to
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
    df = df.dropna(subset=["End_Time"])

    return df

# cleans Severity", "Distance(mi)", "Temperature(F)", "Visibility(mi)", and "Precipitation(in)"
def clean_num_data(df):
    # cleans out data by removing any non numerical data in these collumns
    num_cols = ["Severity", "Distance(mi)", "Temperature(F)", "Visibility(mi)", "Precipitation(in)"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # the actual delete            
    df = df.dropna(subset=["Severity", "Distance(mi)", "Temperature(F)", "Visibility(mi)", "Precipitation(in)"])

    return df

# does all cleans above and cleans "County", "State", "Start_Lat", and "Start_Lng"
def clean_data(df):
    # remove rows if data dont exist
    df = df.dropna(subset=["County", "State", "Start_Lat", "Start_Lng"])
    df = time_data(df)
    df = clean_num_data(df)

    return df
    
# 
def add_data(df):
    # gets hours and makes a new column from start time
    df["hour"] = df["Start_Time"].dt.hour
    # is true (1) when hours are before 6am (exclusive) or after 8pm (inclusive)
    # important when driving because less visability
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)
    # gets which day of the week stored as 0-6
    df["dayofweek"] = df["Start_Time"].dt.dayofweek  
    # checks if day of week is 5(saturday) or 6(sunday)
    # more recreational travel and less morning/evening rush hour
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

def do_data(df):
    clean_data(df)
    add_data(df)
    return df

