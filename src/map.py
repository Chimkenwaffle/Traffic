import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from mpl_toolkits.basemap import Basemap
import geopandas as gpd

def makeMap(df, county_df, title_suffix=""):
    """
    Create a choropleth map of counties using geopandas.
    Uses a smooth green → yellow → red gradient for risk scores.
    """
    
    print("Loading county boundaries...")
    url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip"
    counties_gdf = gpd.read_file(url)
    # Prepare merge fields
    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper().str.strip()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()
    counties_gdf["County_Match"] = counties_gdf["NAME"].str.upper().str.strip()
    # FIPS → State name
    state_fips = {
        '01':'ALABAMA','02':'ALASKA','04':'ARIZONA','05':'ARKANSAS','06':'CALIFORNIA',
        '08':'COLORADO','09':'CONNECTICUT','10':'DELAWARE','11':'DISTRICT OF COLUMBIA',
        '12':'FLORIDA','13':'GEORGIA','15':'HAWAII','16':'IDAHO','17':'ILLINOIS',
        '18':'INDIANA','19':'IOWA','20':'KANSAS','21':'KENTUCKY','22':'LOUISIANA',
        '23':'MAINE','24':'MARYLAND','25':'MASSACHUSETTS','26':'MICHIGAN','27':'MINNESOTA',
        '28':'MISSISSIPPI','29':'MISSOURI','30':'MONTANA','31':'NEBRASKA','32':'NEVADA',
        '33':'NEW HAMPSHIRE','34':'NEW JERSEY','35':'NEW MEXICO','36':'NEW YORK',
        '37':'NORTH CAROLINA','38':'NORTH DAKOTA','39':'OHIO','40':'OKLAHOMA','41':'OREGON',
        '42':'PENNSYLVANIA','44':'RHODE ISLAND','45':'SOUTH CAROLINA','46':'SOUTH DAKOTA',
        '47':'TENNESSEE','48':'TEXAS','49':'UTAH','50':'VERMONT','51':'VIRGINIA',
        '53':'WASHINGTON','54':'WEST VIRGINIA','55':'WISCONSIN','56':'WYOMING','72':'PUERTO RICO'
    }
    counties_gdf["State_Match"] = counties_gdf["STATEFP"].map(state_fips)
    # Merge risk data
    counties = counties_gdf.merge(
        plot_df[["State_Match","County_Match","risk_score","Accidents_Per_1000"]],
        on=["State_Match","County_Match"], how="left"
    )
    # Remove AK, HI, PR for continental US
    counties = counties[~counties["STATEFP"].isin(["02","15","72"])]
    # --- SMOOTH GRADIENT ---
    # Green (low risk) → Yellow (medium risk) → Red (high risk)
    cmap = colors.LinearSegmentedColormap.from_list(
        "risk_gradient", ["#a1d99b", "#fdae61", "#d73027"]
    )
    norm = colors.Normalize(vmin=0, vmax=10)
    # Plot
    fig, ax = plt.subplots(figsize=(18, 10), dpi=60)
    counties.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=1.0)
    counties_with_data = counties[counties["risk_score"].notna()]
    counties_with_data.plot(
        ax=ax,
        column="risk_score",
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.4,
        legend=False
    )
    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Risk Score", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)
    # Zoom to continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(22, 52)
    ax.set_axis_off()
    # Title
    plt.title(
        f"US Accident Risk by County {title_suffix}\n"
        f"{len(plot_df)} counties with data | "
        f"Avg: {plot_df['Accidents_Per_1000'].mean():.2f} accidents per 1000 people 16+",
        fontsize=20, fontweight="bold", pad=20
    )
    plt.tight_layout()
    plt.show()

def makeMapTotal(df, county_df, title_suffix=""):
    url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip"
    counties_gdf = gpd.read_file(url)
    # Prepare merge fields
    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper().str.strip()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()
    counties_gdf["County_Match"] = counties_gdf["NAME"].str.upper().str.strip()
    # FIPS → State name
    state_fips = {
        '01':'ALABAMA','02':'ALASKA','04':'ARIZONA','05':'ARKANSAS','06':'CALIFORNIA',
        '08':'COLORADO','09':'CONNECTICUT','10':'DELAWARE','11':'DISTRICT OF COLUMBIA',
        '12':'FLORIDA','13':'GEORGIA','15':'HAWAII','16':'IDAHO','17':'ILLINOIS',
        '18':'INDIANA','19':'IOWA','20':'KANSAS','21':'KENTUCKY','22':'LOUISIANA',
        '23':'MAINE','24':'MARYLAND','25':'MASSACHUSETTS','26':'MICHIGAN','27':'MINNESOTA',
        '28':'MISSISSIPPI','29':'MISSOURI','30':'MONTANA','31':'NEBRASKA','32':'NEVADA',
        '33':'NEW HAMPSHIRE','34':'NEW JERSEY','35':'NEW MEXICO','36':'NEW YORK',
        '37':'NORTH CAROLINA','38':'NORTH DAKOTA','39':'OHIO','40':'OKLAHOMA','41':'OREGON',
        '42':'PENNSYLVANIA','44':'RHODE ISLAND','45':'SOUTH CAROLINA','46':'SOUTH DAKOTA',
        '47':'TENNESSEE','48':'TEXAS','49':'UTAH','50':'VERMONT','51':'VIRGINIA',
        '53':'WASHINGTON','54':'WEST VIRGINIA','55':'WISCONSIN','56':'WYOMING','72':'PUERTO RICO'
    }
    counties_gdf["State_Match"] = counties_gdf["STATEFP"].map(state_fips)
    # Merge risk data
    counties = counties_gdf.merge(
        plot_df[["State_Match","County_Match","risk_score","Total_Accidents"]],
        on=["State_Match","County_Match"], how="left"
    )
    # Remove AK, HI, PR for continental US
    counties = counties[~counties["STATEFP"].isin(["02","15","72"])]
    # --- SMOOTH GRADIENT ---
    # Green (low risk) → Yellow (medium risk) → Red (high risk)
    cmap = colors.LinearSegmentedColormap.from_list(
        "risk_gradient", ["#a1d99b", "#fdae61", "#d73027"]
    )
    vmax = counties["Total_Accidents"].max() - 65000
    norm = colors.Normalize(vmin=0, vmax=vmax)
    # Plot
    fig, ax = plt.subplots(figsize=(18, 10), dpi=60)
    counties.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=1.0)
    counties_with_data = counties[counties["Total_Accidents"].notna()]
    counties_with_data.plot(
        ax=ax,
        column="Total_Accidents",
        cmap=cmap,
        norm=norm,
        edgecolor="lightgray",
        linewidth=0.4,
        legend=False
    )
    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Risk Score", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)
    # Zoom to continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(22, 52)
    ax.set_axis_off()
    # Title
    plt.title(
        f"US Accident Risk by County {title_suffix}\n"
        f"{len(plot_df)} counties with data | "
        f"Avg: {plot_df['Accidents_Per_1000'].mean():.2f} accidents per 1000 people 16+",
        fontsize=20, fontweight="bold", pad=20
    )
    plt.tight_layout()
    plt.show()
    
    return None
def makeMap2030Total(df, county_df, title_suffix=""):
    url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip"
    counties_gdf = gpd.read_file(url)
    # Prepare merge fields
    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper().str.strip()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()
    counties_gdf["County_Match"] = counties_gdf["NAME"].str.upper().str.strip()
    # FIPS → State name
    state_fips = {
        '01':'ALABAMA','02':'ALASKA','04':'ARIZONA','05':'ARKANSAS','06':'CALIFORNIA',
        '08':'COLORADO','09':'CONNECTICUT','10':'DELAWARE','11':'DISTRICT OF COLUMBIA',
        '12':'FLORIDA','13':'GEORGIA','15':'HAWAII','16':'IDAHO','17':'ILLINOIS',
        '18':'INDIANA','19':'IOWA','20':'KANSAS','21':'KENTUCKY','22':'LOUISIANA',
        '23':'MAINE','24':'MARYLAND','25':'MASSACHUSETTS','26':'MICHIGAN','27':'MINNESOTA',
        '28':'MISSISSIPPI','29':'MISSOURI','30':'MONTANA','31':'NEBRASKA','32':'NEVADA',
        '33':'NEW HAMPSHIRE','34':'NEW JERSEY','35':'NEW MEXICO','36':'NEW YORK',
        '37':'NORTH CAROLINA','38':'NORTH DAKOTA','39':'OHIO','40':'OKLAHOMA','41':'OREGON',
        '42':'PENNSYLVANIA','44':'RHODE ISLAND','45':'SOUTH CAROLINA','46':'SOUTH DAKOTA',
        '47':'TENNESSEE','48':'TEXAS','49':'UTAH','50':'VERMONT','51':'VIRGINIA',
        '53':'WASHINGTON','54':'WEST VIRGINIA','55':'WISCONSIN','56':'WYOMING','72':'PUERTO RICO'
    }
    counties_gdf["State_Match"] = counties_gdf["STATEFP"].map(state_fips)
    # Merge risk data
    counties = counties_gdf.merge(
        plot_df[["State_Match","County_Match","risk_score","Predicted_2030_Accidents_Total"]],
        on=["State_Match","County_Match"], how="left"
    )
    # Remove AK, HI, PR for continental US
    counties = counties[~counties["STATEFP"].isin(["02","15","72"])]
    # --- SMOOTH GRADIENT ---
    # Green (low risk) → Yellow (medium risk) → Red (high risk)
    cmap = colors.LinearSegmentedColormap.from_list(
        "risk_gradient", ["#a1d99b", "#fdae61", "#d73027"]
    )
    vmax = counties["Predicted_2030_Accidents_Total"].max() - 65000
    norm = colors.Normalize(vmin=0, vmax=vmax)
    # Plot
    fig, ax = plt.subplots(figsize=(18, 10), dpi=60)
    counties.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=1.0)
    counties_with_data = counties[counties["Predicted_2030_Accidents_Total"].notna()]
    counties_with_data.plot(
        ax=ax,
        column="Predicted_2030_Accidents_Total",
        cmap=cmap,
        norm=norm,
        edgecolor="lightgray",
        linewidth=0.4,
        legend=False
    )
    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Risk Score", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)
    # Zoom to continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(22, 52)
    ax.set_axis_off()
    # Title
    plt.title(
        f"US Accident Risk by County {title_suffix}\n"
        f"{len(plot_df)} counties with data | "
        f"Avg: {plot_df['Accidents_Per_1000'].mean():.2f} accidents per 1000 people 16+",
        fontsize=20, fontweight="bold", pad=20
    )
    plt.tight_layout()
    plt.show()
    
    return None
def makeMap2030AccidentsPerCapita(df, county_df, title_suffix=""):
    url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip"
    counties_gdf = gpd.read_file(url)
    # Prepare merge fields
    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper().str.strip()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()
    counties_gdf["County_Match"] = counties_gdf["NAME"].str.upper().str.strip()
    # FIPS → State name
    state_fips = {
        '01':'ALABAMA','02':'ALASKA','04':'ARIZONA','05':'ARKANSAS','06':'CALIFORNIA',
        '08':'COLORADO','09':'CONNECTICUT','10':'DELAWARE','11':'DISTRICT OF COLUMBIA',
        '12':'FLORIDA','13':'GEORGIA','15':'HAWAII','16':'IDAHO','17':'ILLINOIS',
        '18':'INDIANA','19':'IOWA','20':'KANSAS','21':'KENTUCKY','22':'LOUISIANA',
        '23':'MAINE','24':'MARYLAND','25':'MASSACHUSETTS','26':'MICHIGAN','27':'MINNESOTA',
        '28':'MISSISSIPPI','29':'MISSOURI','30':'MONTANA','31':'NEBRASKA','32':'NEVADA',
        '33':'NEW HAMPSHIRE','34':'NEW JERSEY','35':'NEW MEXICO','36':'NEW YORK',
        '37':'NORTH CAROLINA','38':'NORTH DAKOTA','39':'OHIO','40':'OKLAHOMA','41':'OREGON',
        '42':'PENNSYLVANIA','44':'RHODE ISLAND','45':'SOUTH CAROLINA','46':'SOUTH DAKOTA',
        '47':'TENNESSEE','48':'TEXAS','49':'UTAH','50':'VERMONT','51':'VIRGINIA',
        '53':'WASHINGTON','54':'WEST VIRGINIA','55':'WISCONSIN','56':'WYOMING','72':'PUERTO RICO'
    }
    counties_gdf["State_Match"] = counties_gdf["STATEFP"].map(state_fips)
    # Merge risk data
    counties = counties_gdf.merge(
        plot_df[["State_Match","County_Match","risk_score","Predicted_2030_Accidents_PerCapita"]],
        on=["State_Match","County_Match"], how="left"
    )
    # Remove AK, HI, PR for continental US
    counties = counties[~counties["STATEFP"].isin(["02","15","72"])]
    # --- SMOOTH GRADIENT ---
    # Green (low risk) → Yellow (medium risk) → Red (high risk)
    cmap = colors.LinearSegmentedColormap.from_list(
        "risk_gradient", ["#a1d99b", "#fdae61", "#d73027"]
    )
    norm = colors.Normalize(vmin=0, vmax=10)
    # Plot
    fig, ax = plt.subplots(figsize=(18, 10), dpi=60)
    counties.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=1.0)
    counties_with_data = counties[counties["Predicted_2030_Accidents_PerCapita"].notna()]
    counties_with_data.plot(
        ax=ax,
        column="Predicted_2030_Accidents_PerCapita",
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.4,
        legend=False
    )
    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Risk Score", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)
    # Zoom to continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(22, 52)
    ax.set_axis_off()
    # Title
    plt.title(
        f"US Accident Risk by County {title_suffix}\n"
        f"{len(plot_df)} counties with data | "
        f"Avg: {plot_df['Accidents_Per_1000'].mean():.2f} accidents per 1000 people 16+",
        fontsize=20, fontweight="bold", pad=20
    )
    plt.tight_layout()
    plt.show()
def makeMap2030AccidentsGrowthPercentage(df, county_df, title_suffix=""):
    url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip"
    counties_gdf = gpd.read_file(url)
    # Prepare merge fields
    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper().str.strip()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()
    counties_gdf["County_Match"] = counties_gdf["NAME"].str.upper().str.strip()
    # FIPS → State name
    state_fips = {
        '01':'ALABAMA','02':'ALASKA','04':'ARIZONA','05':'ARKANSAS','06':'CALIFORNIA',
        '08':'COLORADO','09':'CONNECTICUT','10':'DELAWARE','11':'DISTRICT OF COLUMBIA',
        '12':'FLORIDA','13':'GEORGIA','15':'HAWAII','16':'IDAHO','17':'ILLINOIS',
        '18':'INDIANA','19':'IOWA','20':'KANSAS','21':'KENTUCKY','22':'LOUISIANA',
        '23':'MAINE','24':'MARYLAND','25':'MASSACHUSETTS','26':'MICHIGAN','27':'MINNESOTA',
        '28':'MISSISSIPPI','29':'MISSOURI','30':'MONTANA','31':'NEBRASKA','32':'NEVADA',
        '33':'NEW HAMPSHIRE','34':'NEW JERSEY','35':'NEW MEXICO','36':'NEW YORK',
        '37':'NORTH CAROLINA','38':'NORTH DAKOTA','39':'OHIO','40':'OKLAHOMA','41':'OREGON',
        '42':'PENNSYLVANIA','44':'RHODE ISLAND','45':'SOUTH CAROLINA','46':'SOUTH DAKOTA',
        '47':'TENNESSEE','48':'TEXAS','49':'UTAH','50':'VERMONT','51':'VIRGINIA',
        '53':'WASHINGTON','54':'WEST VIRGINIA','55':'WISCONSIN','56':'WYOMING','72':'PUERTO RICO'
    }
    counties_gdf["State_Match"] = counties_gdf["STATEFP"].map(state_fips)
    # Merge risk data
    counties = counties_gdf.merge(
        plot_df[["State_Match","County_Match","risk_score","Accident_Growth_Percentage"]],
        on=["State_Match","County_Match"], how="left"
    )
    # Remove AK, HI, PR for continental US
    counties = counties[~counties["STATEFP"].isin(["02","15","72"])]
    # --- SMOOTH GRADIENT ---
    # Green (low risk) → Yellow (medium risk) → Red (high risk)
    cmap = colors.LinearSegmentedColormap.from_list(
        "risk_gradient", ["#a1d99b", "#fdae61", "#d73027"]
    )
    norm = colors.Normalize(vmin=0, vmax=100)
    # Plot
    fig, ax = plt.subplots(figsize=(18, 10), dpi=60)
    counties.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=1.0)
    counties_with_data = counties[counties["Accident_Growth_Percentage"].notna()]
    counties_with_data.plot(
        ax=ax,
        column="Accident_Growth_Percentage",
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.4,
        legend=False
    )
    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Risk Score", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)
    # Zoom to continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(22, 52)
    ax.set_axis_off()
    # Title
    plt.title(
        f"US Accident Risk by County {title_suffix}\n"
        f"{len(plot_df)} counties with data | "
        f"Avg: {plot_df['Accidents_Per_1000'].mean():.2f} accidents per 1000 people 16+",
        fontsize=20, fontweight="bold", pad=20
    )
    plt.tight_layout()
    plt.show()