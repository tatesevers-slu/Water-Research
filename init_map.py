import pandas as pd
import folium

# 1. Load your data
df = pd.read_csv("stl_water.csv")

# 2. Split coordinates into lat/lon if needed
df[['Latitude', 'Longitude']] = df['Cordinate (lat/lon)'].str.split(',', expand=True).astype(float)

# 3. List out the columns (elements) you want to visualize
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
    "Managnese (Mn) 257.610",
    "Silver (Ag) 243.778",
    "Gallium (Ga) 417.206",
    "Thallium (Tl) 190.801",
    "Vandium (V) 290.880",
    "Indium (In) 230.606",
    "Uranium (U) 385.958",
    "Uranium (U) 367.007",
    "Uranium (U) 409.014",
    "Uranium (U) 424.167",
    "Selenium (Se) 196.026"
]
# Ensure each element column is numeric if possible
# "ND" or other non-numeric become NaN
for el in elements:
    df[el] = pd.to_numeric(df[el], errors="coerce")

# 4. Precompute min & max for each element (ignoring NaN)
element_stats = {}
for el in elements:
    valid_values = df[el].dropna()
    if not valid_values.empty:
        element_stats[el] = (valid_values.min(), valid_values.max())
    else:
        # If no valid values, default min & max to 0
        element_stats[el] = (0, 0)

# A helper function to scale a value into a chosen marker-size range
def scale_marker(value, min_val, max_val, min_radius=3, max_radius=15):
    """Linearly maps value in [min_val, max_val] to [min_radius, max_radius]."""
    if pd.isna(value):
        return 0
    if max_val == min_val:  # avoid div-by-zero if all values are the same
        return (min_radius + max_radius) / 2
    return min_radius + (value - min_val) * (max_radius - min_radius) / (max_val - min_val)

# 5. Create the main Folium map
m = folium.Map(location=[38.6270, -90.1994], zoom_start=11)

# 6. For each element, create a FeatureGroup (off by default) and add CircleMarkers
for el in elements:
    fg = folium.FeatureGroup(name=el, show=False)  # <--- show=False makes layer unchecked initially
    
    min_val, max_val = element_stats[el]

    for _, row in df.iterrows():
        lat = row["Latitude"]
        lon = row["Longitude"]
        value = row[el]
        
        # Skip if lat/lon is missing
        if pd.isna(lat) or pd.isna(lon):
            continue
        
        # Handle the case of NaN (e.g., ND)
        if pd.isna(value):
            popup_text = f"{row['Name']}<br>{el}: ND"
            radius = 0
        else:
            popup_text = f"{row['Name']}<br>{el}: {value:.2f}"
            # Compute a normalized circle size
            radius = scale_marker(value, min_val, max_val)
        
        # Optionally skip drawing if radius=0
        if radius == 0:
            continue
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            popup=popup_text,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6
        ).add_to(fg)
    
    fg.add_to(m)

# 7. Add a LayerControl (to toggle layers on/off)
folium.LayerControl().add_to(m)

# 8. Save the map to HTML
m.save("stl_water_sampling_map_all_elements.html")
