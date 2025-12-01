import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataCleaning import get_csv
from dataCleaning import do_traffic_data
from dataCleaning import do_driver_data
from map import makeMap
from train import clean
from train import train

def main():
    # Load and clean accident data
    df = get_csv("sobhanmoosavi/us-accidents", "/US_Accidents_March23.csv")
    df = do_traffic_data(df)
       
    # Filter for year 2020
    df2020 = df[df["Start_Time"].dt.year == 2020]

    # Skip the header rows and use proper column names
    drivers_df = pd.read_excel("TRAFFIC/data/counties-agegroup-2020.xlsx", skiprows=5)
    drivers_df = do_driver_data(drivers_df)
    
    # Define grouping and aggregation
    group_cols = ["State", "County"]
    agg_dict = {
        "ID": "count",
        "Severity": "mean",
        "Distance(mi)": "mean",
        "Temperature(F)": "mean",
        "Visibility(mi)": "mean",
        "Precipitation(in)": "mean",
        "Is_Night": "mean",
        "Is_Weekend": "mean"
    }

    county_df, feature_cols = clean(df, group_cols, agg_dict)

    county_df2020, feature_cols2020= clean(df2020, group_cols, agg_dict)
    
    # ============================================================
    # Merge with driver data (people 16+)
    # ============================================================

    # Standardize county names for better matching
    county_df['County_Clean'] = county_df['County'].str.strip().str.replace(' County', '', regex=False)
    drivers_df['County_Clean'] = drivers_df['County'].str.strip().str.replace(' County', '', regex=False)
    
    county_df['State_Clean'] = county_df['State'].str.strip()
    drivers_df['State_Clean'] = drivers_df['State'].str.strip()
    
    county_df = county_df.merge(
        drivers_df[['State_Clean', 'County_Clean', 'Total_People_16_plus']], 
        left_on=['State_Clean', 'County_Clean'],
        right_on=['State_Clean', 'County_Clean'],
        how='left'
    )

    county_df = county_df.dropna(subset=["Total_People_16_plus"])
    
    # Same for 2020
    county_df2020['County_Clean'] = county_df2020['County'].str.strip().str.replace(' County', '', regex=False)
    county_df2020['State_Clean'] = county_df2020['State'].str.strip()
    
    county_df2020 = county_df2020.merge(
        drivers_df[['State_Clean', 'County_Clean', 'Total_People_16_plus']], 
        left_on=['State_Clean', 'County_Clean'],
        right_on=['State_Clean', 'County_Clean'],
        how='left'
    )
    county_df2020 = county_df2020.dropna(subset=["Total_People_16_plus"])
    
    # ============================================================
    # Calculate accidents per capita (per 1000 people 16+)
    # ============================================================
    
    county_dfTotal = county_df
    if len(county_df) > 0:
        county_df["Accidents_Per_1000"] = (county_df["Total_Accidents"] / county_df["Total_People_16_plus"]) * 100

    if len(county_df2020) > 0:
        county_df2020["Accidents_Per_1000"] = (county_df2020["Total_Accidents"] / county_df2020["Total_People_16_plus"]) * 100
    
    # Add population as a feature
    feature_cols_extended = feature_cols + ["Total_People_16_plus"]
    
    # ============================================================
    # Train-test split - predict accidents per capita
    # ============================================================

    X = county_df[feature_cols_extended].values
    y = county_df["Accidents_Per_1000"].values
    XTotal = county_dfTotal[feature_cols_extended].values
    yTotal = county_dfTotal["Total_Accidents"].values
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, county_df.index, test_size=0.2, random_state=42
    )
    XTotal_train, XTotal_test, yTotal_train, yTotal_test, idx_train, idx_test = train_test_split(
        XTotal, yTotal, county_df.index, test_size=0.2, random_state=42
    )
    
    # ============================================================
    # Train Random Forest
    # ============================================================
    rf = train(200, 42, -1, X_train, y_train, X_test, y_test)
    rfTotal = train(200, 42, -1, XTotal_train, yTotal_train, XTotal_test, yTotal_test)
    # ============================================================
    # Predict risk (based on accidents per capita)
    # ============================================================
    
    all_preds = rf.predict(X)
    total_preds = rfTotal.predict(X)
    min_pred = all_preds.min()
    max_pred = all_preds.max()
    minTotal_pred = total_preds.min()
    maxTotal_pred = total_preds.max()
    county_df["risk_score"] = 100 * (all_preds - min_pred) / (max_pred - min_pred + 1e-9)
    county_dfTotal["risk_score"] = 100 * (all_preds - min_pred) / (max_pred - min_pred + 1e-9)
    # Same for 2020 data (only if we have data)
    if len(county_df2020) > 0:
        X_2020 = county_df2020[feature_cols_extended].values
        all_preds_2020 = rf.predict(X_2020)
        min_pred_2020 = all_preds_2020.min()
        max_pred_2020 = all_preds_2020.max()
        county_df2020["risk_score"] = 100 * (all_preds_2020 - min_pred_2020) / (max_pred_2020 - min_pred_2020 + 1e-9)
    else:
        print("Skipping 2020 risk scores - no 2020 data available")
    
    # ============================================================
    # Feature Importance
    # ============================================================

    feature_importance = pd.DataFrame({
        'feature': feature_cols_extended,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    
    # ============================================================
    # Top counties by accidents per capita
    # ============================================================
    print("\n=== Top 20 High-Risk Counties (by accidents per 1000 people 16+) ===")
    top_n = 20
    top_risk = county_df.sort_values("Accidents_Per_1000", ascending=False).head(top_n)
    print(top_risk[["State", "County", "Total_Accidents", "Total_People_16_plus", "Accidents_Per_1000", "risk_score"]])
    
    # ============================================================
    # Bottom counties (safest)
    # ============================================================
    print("\n=== Bottom 20 Safest Counties (by accidents per 1000 people 16+) ===")
    bottom_risk = county_df.sort_values("Accidents_Per_1000", ascending=True).head(top_n)
    print(bottom_risk[["State", "County", "Total_Accidents", "Total_People_16_plus", "Accidents_Per_1000", "risk_score"]])
    
    # ============================================================
    # Plot US bubble maps
    # ============================================================
    print("\n=== Generating Maps ===")
    makeMap(df, county_df, title_suffix="(All Years)")
    
    if len(county_df2020) > 0 and len(df2020) > 0:
        makeMap(df2020, county_df2020, title_suffix="(2020 Only)")
    else:
        print("Skipping 2020 map - insufficient 2020 data")

    makeMap(df, county_dfTotal, title_suffix="Total AAAAAAAA")

if __name__ == "__main__":
    main()
    