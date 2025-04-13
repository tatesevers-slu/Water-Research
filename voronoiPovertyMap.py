import folium
import branca.colormap as cm
import pandas as pd
import numpy as np

# =============================================================================
# PART 1: Process the Income Data and Add the Voronoi Layer
# =============================================================================

# Load median household income data
df_income = pd.read_csv("Median_Household_Income.csv")
df_income["INTPTLAT"] = pd.to_numeric(df_income["INTPTLAT"], errors="coerce")
df_income["INTPTLON"] = pd.to_numeric(df_income["INTPTLON"], errors="coerce")
df_income = df_income.dropna(subset=["INTPTLAT", "INTPTLON", "Median_Household_Income"])

# Set up a color scale for income (red = low, green = high)
min_income = df_income["Median_Household_Income"].min()
max_income = df_income["Median_Household_Income"].max()
colormap_income = cm.LinearColormap(
    colors=["red", "yellow", "green"],
    vmin=min_income,
    vmax=max_income,
    caption="Median Household Income"
)

# Initialize the base Folium map (centered on St. Louis)
map_center = [38.6270, -90.1994]
m = folium.Map(location=map_center, zoom_start=10)

# Add the income Voronoi GeoJSON layer (previously generated and saved as "income_voronoi.geojson")
def income_voronoi_style(feature):
    income = feature["properties"]["Median_Household_Income"]
    fill_color = colormap_income(income) if income is not None else "#999999"
    return {
        "stroke": False,       # Remove boundaries between shapes
        "weight": 0,
        "fillOpacity": 0.3,    # Lower opacity for translucency
        "fillColor": fill_color,
    }

folium.GeoJson(
    "income_voronoi.geojson",
    name="Income Voronoi",
    style_function=income_voronoi_style,
    tooltip=folium.GeoJsonTooltip(
        fields=["Median_Household_Income"],
        aliases=["Median Household Income:"]
    )
).add_to(m)

colormap_income.add_to(m)
folium.LayerControl().add_to(m)

# =============================================================================
# PART 2: Process Water Sampling Data and Compute the Overall Contamination Index
# =============================================================================

# Load water sampling data and split coordinates into Latitude and Longitude
df_water = pd.read_csv("stl_water.csv")
df_water[['Latitude', 'Longitude']] = df_water['Cordinate (lat/lon)'].str.split(',', expand=True).astype(float)

# List of contaminant element columns (as provided)
elements = [
    "Arsenic (As) 193.696",
    "Cadmium (Cd) 214.440",
    "Chromium (Cr) 267.716",
    "Copper (Cu) 324.752",
    "Lead (Pb) 220.353",
    "Nickel (Ni) 231.604",
    "Selenium (Se) 196.026",
    "Silver (Ag) 243.778",
    "Zinc (Zn) 213.857",
    "Lithium (Li) 670.784",
    "Sodium (Na) 589.592",
    "Potassium (K) 766.490",
    "Magnesium (Mg) 285.213",
    "Calcium (Ca) 317.933",
    "Strontium (Sr) 421.552",
    "Beryllium (Be) 313.107",
    "Ceseium (Cs) 455.531",
    "Aluminum (Al) 396.153",
    "Barium (Ba) 233.527",
    "Cobalt (Co) 228.616",
    "Iron (Fe) 238.204",
    "Manganese (Mn) 259.372",
    "Manganese (Mn) 257.610",
    "Gallium (Ga) 417.206",
    "Thallium (Tl) 190.801",
    "Vandium (V) 290.880",
    "Indium (In) 230.606",
    "Uranium (U) 385.958",
    "Uranium (U) 367.007",
    "Uranium (U) 409.014",
    "Uranium (U) 424.167"
]

# Convert each element column to numeric (non-numeric entries become NaN)
for el in elements:
    df_water[el] = pd.to_numeric(df_water[el], errors="coerce")

# Normalize each contaminant column using min–max scaling so that each ranges from 0 to 1.
norm_cols = []
for el in elements:
    norm_col = f"Norm_{el}"
    col_min = df_water[el].min()
    col_max = df_water[el].max()
    if col_max != col_min:
        df_water[norm_col] = (df_water[el] - col_min) / (col_max - col_min)
    else:
        df_water[norm_col] = 0.0  # If no variation, set normalized value to 0.
    norm_cols.append(norm_col)

# Compute an overall contamination index as the simple average of the normalized contaminant values.
df_water['Overall_Contamination_Index'] = df_water[norm_cols].mean(axis=1, skipna=True)

# (Optional) Print summary statistics to review the normalized ranges
print("Overall Contamination Index summary:")
print(df_water['Overall_Contamination_Index'].describe())

# =============================================================================
# PART 3: Overlay Water Sampling Points with the Overall Contamination Index
# =============================================================================

# Set up a color scale for the contamination index.
min_cont = df_water['Overall_Contamination_Index'].min()
max_cont = df_water['Overall_Contamination_Index'].max()
cont_colormap = cm.LinearColormap(
    colors=["blue", "cyan"],   # Blue for lower index, cyan for higher index; adjust as desired.
    vmin=min_cont,
    vmax=max_cont,
    caption="Overall Contamination Index"
)
cont_colormap.add_to(m)

# Plot each water sample as a CircleMarker using the contamination index to color the marker.
for idx, row in df_water.iterrows():
    lat, lon = row["Latitude"], row["Longitude"]
    overall_index = row["Overall_Contamination_Index"]
    marker_color = cont_colormap(overall_index)
    popup_text = f"Overall Contamination Index: {overall_index:.2f}"
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=marker_color,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.7,
        popup=popup_text
    ).add_to(m)

# =============================================================================
# Save the Final Map
# =============================================================================

m.save("income_voronoi_water_map.html")
print("Interactive map saved as 'income_voronoi_water_map.html'.")
