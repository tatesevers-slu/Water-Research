import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

#load
df = pd.read_csv("stl_water.csv")
#split coordinate string into separate Latitude and Longitude columns
df[['Latitude', 'Longitude']] = df['Cordinate (lat/lon)'].str.split(',', expand=True).astype(float)

elements = [
    "Copper (Cu) 324.752", "Selenium (Se) 196.026", "Silver (Ag) 243.778", "Zinc (Zn) 213.857", 
    "Lithium (Li) 670.784", "Sodium (Na) 589.592", "Potassium (K) 766.490", "Magnesium (Mg) 285.213", 
    "Calcium (Ca) 317.933", "Strontium (Sr) 421.552", "Ceseium (Cs) 455.531", "Barium (Ba) 233.527", 
    "Iron (Fe) 238.204", "Manganese (Mn) 259.372", "Manganese (Mn) 257.610", "Thallium (Tl) 190.801", 
    "Vandium (V) 290.880", "Uranium (U) 385.958", "Selenium (Se) 196.026"
]

# "ND" become NaN, otherwise numeric sample values
for el in elements:
    df[el] = pd.to_numeric(df[el], errors="coerce")

# missing values filled with column mean
# that being average contaminant across all samples
df_elements = df[elements]
df_elements_imputed = df_elements.fillna(df_elements.mean())

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_elements_imputed)

# PCA step
pca = PCA(n_components=6)
pca_scores = pca.fit_transform(X_scaled)
pca_columns = [f'PC{i+1}' for i in range(pca_scores.shape[1])]
pca_df = pd.DataFrame(pca_scores, columns=pca_columns)
df = pd.concat([df, pca_df], axis=1)

# plot variance ratio
plt.figure(figsize=(6, 4))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        tick_label=[f'PC{i}' for i in range(1, len(pca.explained_variance_ratio_) + 1)])
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.tight_layout()
plt.show()

# load and clean the median income data
# assign the nearest income with KDtree
income_df = pd.read_csv("Median_Household_Income.csv")
income_df["INTPTLAT"] = pd.to_numeric(income_df["INTPTLAT"], errors="coerce")
income_df["INTPTLON"] = pd.to_numeric(income_df["INTPTLON"], errors="coerce")
# Clean and convert the median income column
income_df["Median_Household_Income"] = income_df["Median_Household_Income"].replace('[\$,]', '', regex=True)
income_df["Median_Household_Income"] = pd.to_numeric(income_df["Median_Household_Income"], errors="coerce")

# Build a KDTree using the income data coordinates
income_coords = income_df[["INTPTLAT", "INTPTLON"]].values
tree = cKDTree(income_coords)

def get_nearest_income(lat, lon):
    distance, idx = tree.query([lat, lon])
    return income_df.iloc[idx]["Median_Household_Income"]

# For each water sample, attach the nearest median household income
df["Nearest_Median_Income"] = df.apply(lambda row: get_nearest_income(row["Latitude"], row["Longitude"]), axis=1)
print("Nearest Median Income (first 5):")
print(df["Nearest_Median_Income"].head())

# regression of median household income
# Drop rows with missing values (after searching for median household income)
df_reg = df.dropna(subset=["Nearest_Median_Income", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6"])
print("Number of observations for regression:", df_reg.shape[0])

X_reg = df_reg[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']]
X_reg = sm.add_constant(X_reg)
#x_reg formed using the first 6 principal components
y = df_reg['Nearest_Median_Income']
#y is dependent var

model = sm.OLS(y, X_reg).fit()
print(model.summary())

plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid(True)
plt.show()
