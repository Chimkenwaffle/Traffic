import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import map
from dataCleaning import get_csv
from dataCleaning import do_traffic_data
from dataCleaning import do_driver_data
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
# Predict 2030 using Linear Regression
# ============================================================

    county_df["Projected_2030_Population_16_plus"] = (
        county_df["Total_People_16_plus"] + 
        county_df["Total_Decade_Population_Change"]
    )

    # Make sure projected population is not negative
    county_df["Projected_2030_Population_16_plus"] = county_df["Projected_2030_Population_16_plus"].clip(lower=0)

    print("Training Linear Regression model for 2030 predictions...")

    # Create features for training (using 2020 data)
    X_train_lr = np.column_stack([
        county_df[feature_cols].values,
        county_df["Total_People_16_plus"].values
    ])

    # Target: accidents per capita in 2020
    y_train_lr = county_df["Accidents_Per_1000"].values

    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, y_train_lr)

    # Check model performance on training data
    y_pred_train_lr = lr_model.predict(X_train_lr)
    print(f"Linear Regression R² on training data: {r2_score(y_train_lr, y_pred_train_lr):.4f}")
    print(f"Linear Regression MAE on training data: {mean_absolute_error(y_train_lr, y_pred_train_lr):.4f}")

    # 3. Create features for 2030 prediction
    # Use same environmental features but with projected population
    X_2030 = np.column_stack([
        county_df[feature_cols].values,
        county_df["Projected_2030_Population_16_plus"].values
    ])

    # Predict per-capita accidents for 2030 using Linear Regression
    # This gives us accidents per 1000 people
    predicted_per_capita_2030 = lr_model.predict(X_2030)

    # Ensure predictions are non-negative
    predicted_per_capita_2030 = np.maximum(predicted_per_capita_2030, 0)

    # Store the per-capita prediction
    county_df["Predicted_2030_Accidents_PerCapita"] = predicted_per_capita_2030

    # Convert to total accidents
    # Formula: (accidents per 1000) * (population / 1000)
    county_df["Predicted_2030_Accidents_Total"] = (
        predicted_per_capita_2030 * 
        (county_df["Projected_2030_Population_16_plus"] / 1000)
    )

    # 4. Calculate growth metrics
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

    # 5. Print Linear Regression coefficients for interpretation
    print("\n=== Linear Regression Coefficients ===")
    feature_names_extended = feature_cols + ["Total_People_16_plus"]
    coef_df = pd.DataFrame({
        'Feature': feature_names_extended,
        'Coefficient': lr_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    print(coef_df)
    print(f"\nIntercept: {lr_model.intercept_:.4f}")
    
    # ============================================================
    # Plot US bubble map
    # ============================================================
    print("\n=== Generating Map ===")
    map.makeMap(df, county_df, title_suffix="")
    map.makeMapTotal(df, county_df, title_suffix="")
    map.makeMap2030Total(df, county_df, title_suffix="")
    map.makeMap2030AccidentsGrowthPercentage(df, county_df, title_suffix="")
    map.makeMap2030PopulationGrowthPercentage(df, county_df, title_suffix="")
    map.makeMap2030PopulationGrowth(df, county_df, title_suffix="")
    map.makeStateMapTotal(county_df, title_suffix="")
    map.makeStateMap2030Total(county_df, title_suffix="")
    map.makeStateMapAccidentGrowth(county_df, title_suffix="")
    map.makeStateMapPopulationGrowth(county_df, title_suffix="")
if __name__ == "__main__":
    main()
    