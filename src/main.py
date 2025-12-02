import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dataCleaning import get_csv
from dataCleaning import do_traffic_data
from dataCleaning import do_driver_data
from map import makeMap
from map import makeMapTotal
from map import makeMap2030Total
from map import makeMap2030AccidentsPerCapita
from map import makeMap2030AccidentsGrowthPercentage
from train import clean
from train import train
from train import predictRisk

def main():
    # Load and clean accident data
    df = get_csv("sobhanmoosavi/us-accidents", "/US_Accidents_March23.csv")
    df = do_traffic_data(df)
       
    # Filter for year 2020 ONLY
    df = df[df["Start_Time"].dt.year == 2020]
    print(f"Using 2020 data only: {len(df)} accidents")

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
    
    # ============================================================
    # Merge with driver data (people 16+)
    # ============================================================

    # Standardize county names for better matching
    county_df['County_Clean'] = county_df['County'].str.strip().str.replace(' County', '', regex=False)
    drivers_df['County_Clean'] = drivers_df['County'].str.strip().str.replace(' County', '', regex=False)
    
    county_df['State_Clean'] = county_df['State'].str.strip()
    drivers_df['State_Clean'] = drivers_df['State'].str.strip()
    
    county_df = county_df.merge(
        drivers_df[['State_Clean', 'County_Clean', 'Total_People_16_plus', 'Total_Decade_Population_Change']], 
        left_on=['State_Clean', 'County_Clean'],
        right_on=['State_Clean', 'County_Clean'],
        how='left'
    )

    county_df = county_df.dropna(subset=["Total_People_16_plus", "Total_Decade_Population_Change"])
    
    # ============================================================
    # Calculate accidents per capita (per 1000 people 16+)
    # ============================================================
    
    if len(county_df) > 0:
        county_df["Accidents_Per_1000"] = (county_df["Total_Accidents"] / county_df["Total_People_16_plus"]) * 1000
    
    # Add population as a feature
    feature_cols_extended = feature_cols + ["Total_People_16_plus"]
    
    # ============================================================
    # Train-test split - predict accidents per capita
    # ============================================================

    X = county_df[feature_cols_extended].values
    y = county_df["Accidents_Per_1000"].values
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, county_df.index, test_size=0.2, random_state=42
    )
    
    # ============================================================
    # Train Random Forest
    # ============================================================
    print("\n=== Training Per-Capita Accident Model (2020 Data) ===")
    rf = train(200, 42, -1, X_train, y_train, X_test, y_test)
    
    # ============================================================
    # Predict risk (based on accidents per capita)
    # ============================================================
    print("\n=== Predicting Risk Scores ===")
    predictRisk(rf, county_df, X)
    
    # ============================================================
    # Feature Importance
    # ============================================================

    feature_importance = pd.DataFrame({
        'feature': feature_cols_extended,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== Feature Importance (Per-Capita Model) ===")
    print(feature_importance)
    
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
    # Predict 2030
    # ============================================================
    
    print("\n=== Predicting 2030 Accident Rates ===")

# Check if we have the population change column

    # 1. Project 2030 population using decade change
    # Total_Decade_Population_Change is the ABSOLUTE change from 2010-2020
    # We'll apply the same absolute change for 2020-2030
    county_df["Projected_2030_Population_16_plus"] = (
        county_df["Total_People_16_plus"] + 
        county_df["Total_Decade_Population_Change"]
    )
    
    # Make sure projected population is not negative
    county_df["Projected_2030_Population_16_plus"] = county_df["Projected_2030_Population_16_plus"].clip(lower=0)
    
    print("Training 2030 prediction model...")
    
    # 2. Create features for 2030 prediction
    # Use same environmental features but with projected population
    X_2030 = np.column_stack([
        county_df[feature_cols].values,
        county_df["Projected_2030_Population_16_plus"].values
    ])
    
    # Predict per-capita accidents for 2030 using the trained RF model
    # This gives us accidents per 1000 people
    predicted_per_capita_2030 = rf.predict(X_2030)
    
    # Store the per-capita prediction
    county_df["Predicted_2030_Accidents_PerCapita"] = predicted_per_capita_2030
    
    # Convert to total accidents
    # Formula: (accidents per 1000) * (population / 1000)
    county_df["Predicted_2030_Accidents_Total"] = (
        predicted_per_capita_2030 * 
        (county_df["Projected_2030_Population_16_plus"] / 1000)
    )
    
    # 3. Calculate growth metrics
    county_df["Accident_Growth_2020_to_2030"] = (
        county_df["Predicted_2030_Accidents_Total"] - county_df["Total_Accidents"]
    )
    
    county_df["Accident_Growth_Percentage"] = (
        (county_df["Accident_Growth_2020_to_2030"] / county_df["Total_Accidents"]) * 100
    )
    
    county_df["Population_Growth_2020_to_2030"] = (
        county_df["Projected_2030_Population_16_plus"] - county_df["Total_People_16_plus"]
    )
    
    county_df["Population_Growth_Percentage"] = (
        (county_df["Population_Growth_2020_to_2030"] / county_df["Total_People_16_plus"]) * 100
    )
    
    # ============================================================
    # Plot US bubble map
    # ============================================================
    print("\n=== Generating Map ===")
    print(df.columns)
    makeMap(df, county_df, title_suffix="(2020 Only)")
    makeMapTotal(df, county_df, title_suffix="(2020 Total Accidents)")
    makeMap2030Total(df, county_df, title_suffix="(2030 Prediction Total Accidents)")
    makeMap2030AccidentsPerCapita(df, county_df, title_suffix="(2030 Prediction Accidents Per Capita)")
    makeMap2030AccidentsGrowthPercentage(df, county_df, title_suffix="(2030 Prediction Growth of Accidents)")
if __name__ == "__main__":
    main()
    