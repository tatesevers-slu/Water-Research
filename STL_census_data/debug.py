import pandas as pd
import os

script_dir = "C:/Users/Tate Severs/Water-Research/STL_census_data"
csv_file = os.path.join(script_dir, "Watershed_bnd.csv")

df = pd.read_csv(csv_file)
print(df.head())  # Show the first few rows
print(df.columns)  # Show column names
