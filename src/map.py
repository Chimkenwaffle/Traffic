import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
        
def makeMap(df, county_df, title_suffix=""):
    """
    Create a choropleth map of counties using geopandas.
    Falls back to Basemap scatter plot if geopandas is missing or fails.
    """
    try:

        # Load county boundaries
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

        # Remove AK, HI, PR
        counties = counties[~counties["STATEFP"].isin(["02","15","72"])]

        # Plot
        fig, ax = plt.subplots(figsize=(18, 10), dpi=25)

        vmax = np.percentile(plot_df["risk_score"], 99)
        vmin = 0

        # Draw background counties (no data)
        counties.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=0.5)

        # Draw counties with data
        counties_with_data = counties[counties["risk_score"].notna()]
        counties_with_data.plot(
            ax=ax,
            column="risk_score",
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
            edgecolor="black",
            linewidth=0.4,
            legend=False
        )

        # Colorbar
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap="Reds")
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
        cbar.set_label("Risk Score (0–100)", fontsize=16, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)

        # Auto-zoom to continental US (prevents Florida cutoff)
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

        try:
            manager = plt.get_current_fig_manager()
            manager.window.state("zoomed")
        except:
            pass

        plt.show()
        print(f"Successfully plotted {len(counties_with_data)} counties.")

    except Exception as e:
        print("Geopandas failed:", e)
        print("Using fallback Basemap scatter instead...")
        makeMap_fallback(df, county_df, title_suffix)



def makeMap_fallback(df, county_df, title_suffix=""):
    """
    Fallback version that draws circles on a Basemap projection.
    Used only when geopandas is unavailable.
    """
    county_coords = df.groupby(["State","County"]).agg({
        "Start_Lat":"mean",
        "Start_Lng":"mean"
    }).reset_index()

    plot_df = county_df.merge(county_coords, on=["State","County"])
    vmax = np.percentile(plot_df["risk_score"], 99)

    plt.figure(figsize=(28, 18))
    m = Basemap(
        llcrnrlon=-128, llcrnrlat=22,
        urcrnrlon=-65,  urcrnrlat=50,
        projection="lcc", lat_1=33, lat_2=45, lon_0=-95
    )

    m.drawcoastlines(linewidth=1)
    m.drawstates(linewidth=1)
    m.drawcountries(linewidth=1)

    norm = colors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.Reds

    x, y = m(plot_df["Start_Lng"].values, plot_df["Start_Lat"].values)
    risks = plot_df["risk_score"].values

    m.scatter(
        x, y,
        c=cmap(norm(risks)),
        s=3000,
        alpha=0.7,
        edgecolors="black",
        linewidths=1.3,
        zorder=5
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.7, pad=0.04)
    cbar.set_label("Risk Score (0–100)", fontsize=16, fontweight="bold")

    plt.title(
        f"US Accident Risk by County (Fallback) {title_suffix}\n"
        f"{len(plot_df)} counties | Avg: {plot_df['Accidents_Per_1000'].mean():.2f} per 1000",
        fontsize=20, fontweight="bold"
    )

    plt.tight_layout()
    plt.show()
    
