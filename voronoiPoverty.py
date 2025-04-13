# voronoi_income_interpolation.py

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas as gpd

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions into finite polygons.
    
    Parameters:
        vor (scipy.spatial.Voronoi): Voronoi diagram.
        radius (float, optional): Distance to 'close' infinite regions.
        
    Returns:
        new_regions (list): List of regions where each region is a list of vertices indices.
        new_vertices (ndarray): Array of vertices coordinates.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    # Determine a distance to 'close' infinite regions
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Map each point to its ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct each region
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        # For non-finite regions, rebuild the region.
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue  # both endpoints are finite
            # Calculate the missing endpoint.
            t = vor.points[p2] - vor.points[p1]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # Order vertices in counterclockwise order
        vs = np.array([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.array(new_vertices)

def main():
    # -------------------------
    # 1. Load and clean the income data
    # -------------------------
    df_income = pd.read_csv("Median_Household_Income.csv")
    df_income["INTPTLAT"] = pd.to_numeric(df_income["INTPTLAT"], errors="coerce")
    df_income["INTPTLON"] = pd.to_numeric(df_income["INTPTLON"], errors="coerce")
    df_income = df_income.dropna(subset=["INTPTLAT", "INTPTLON", "Median_Household_Income"])

    # -------------------------
    # 2. Extract points (using [longitude, latitude] order)
    # -------------------------
    points = df_income[["INTPTLON", "INTPTLAT"]].values

    # -------------------------
    # 3. Compute the Voronoi tessellation
    # -------------------------
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # -------------------------
    # 4. Define a bounding box (from the min/max of the income points) and clip regions.
    # -------------------------
    min_x, min_y = points[:, 0].min(), points[:, 1].min()
    max_x, max_y = points[:, 0].max(), points[:, 1].max()
    bbox = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])

    voronoi_polys = []
    incomes = []
    for i, region in enumerate(regions):
        polygon_points = vertices[region]
        poly = Polygon(polygon_points)
        # Clip polygon to the bounding box to avoid oversized shapes.
        poly = poly.intersection(bbox)
        voronoi_polys.append(poly)
        incomes.append(df_income.iloc[i]["Median_Household_Income"])

    # -------------------------
    # 5. Build a GeoDataFrame and export to GeoJSON
    # -------------------------
    gdf = gpd.GeoDataFrame({
        "Median_Household_Income": incomes,
        "geometry": voronoi_polys
    }, crs="EPSG:4326")

    output_geojson = "income_voronoi.geojson"
    gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"GeoJSON file saved as {output_geojson}")

    # -------------------------
    # 6. Optional: Plotting the results for verification
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(column="Median_Household_Income", cmap="viridis", edgecolor="black",
             legend=True, ax=ax)
    ax.scatter(points[:, 0], points[:, 1], color="red", marker="x", label="Income Points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Voronoi Diagram of Median Household Income")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
