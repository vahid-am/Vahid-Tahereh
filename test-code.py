import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Create fake dataset
#data = {
 #   "age": [25, 30, None, 40, 35],
  #  "salary": [50000, 60000, 55000, None, 65000]
#}

#df = pd.DataFrame(data)

# Clean missing values
#df["age"].fillna(df["age"].median(), inplace=True)
#df["salary"].fillna(df["salary"].median(), inplace=True)

#print(df.describe())

#print(df.shape())

# Use the raw GitHub URL
data_path = 'https://raw.githubusercontent.com/vahid-am/Vahid-Tahereh/main/case-data.csv'
data = pd.read_csv(data_path)

print(data.head())
print(data.shape)

# Checking for missining values 
missing = pd.DataFrame({
    "Missing Count": data.isnull().sum(),
    "Missing %": (data.isnull().sum() / len(data)) * 100
})

print(missing)

total_missing = data.isnull().sum().sum()
total_values = data.size

percentage_missing = (total_missing / total_values) * 100
print("Total Missing %:", percentage_missing)

# Cleaning and seperating numerical and categorical data
data = data.drop(columns=['Unnamed: 0'])
data['clicks'] = data['clicks'].fillna(0)  # Alternatively, we can use the median or drop 'clicks,' as it does not play any role in the following modeling.

#  Pivot weekly-channel costs
weekly = data.groupby(['week', 'channel']).agg({'cost':'sum', 'applications':'max'}).reset_index()
cost_wide = weekly.pivot(index='week', columns='channel', values='cost').fillna(0)
applications = weekly.groupby('week')['applications'].max()
df_model = cost_wide.copy()
df_model['applications'] = applications
df_model['week_index'] = np.arange(len(df_model)) 

print(df_model.head())
print(data.shape)

# ==========================================
# Prepare Data
# ==========================================
df = df_model.copy()

# Ensure datetime index and sorted order
df.index = pd.to_datetime(df.index)
df = df.sort_index()

features = ['affiliates', 'facebook', 'google', 'tv', 'week_index']
target = 'applications'

X = df[features]
y = df[target]

# ==========================================
# Train/Test Split (Time-based)
# ==========================================
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# ==========================================
# Fit Linear Regression Model
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# Model Results
# ==========================================
print("Intercept:", model.intercept_)
print("\nCoefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef}")

# ==========================================
# Predictions & Evaluation
# ==========================================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nTest Performance")
print("RMSE:", rmse)
print("R²:", r2)

# ==========================================
# Plot Actual vs Predicted
# ==========================================
plt.figure()
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, linestyle='--', label='Predicted')
plt.title("Actual vs Predicted Applications")
plt.xticks(rotation=45)
plt.legend()

plt.savefig("actual_vs_predicted.png")  # saves plot to a file
plt.close()  # closes the figure to free memory

# ==========================================
# Normalized Spend vs Coefficient
# ==========================================
coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_
})
coef_df_plot = coef_df[coef_df['feature'] != 'week_index']
cost_totals = df[['affiliates', 'facebook', 'google', 'tv']].sum()

comparison_df = pd.DataFrame({
    'Total Spend': cost_totals,
    'Coefficient': coef_df_plot.set_index('feature')['coefficient']
})

comparison_df = comparison_df / comparison_df.abs().max()

plt.figure()
comparison_df.plot(kind='bar')
plt.title("Normalized Spend vs Coefficient")
plt.xticks(rotation=45)

plt.savefig("Normalized Spend vs Coefficient.png")  # saves plot to a file
plt.close()  # closes the figure to free memory
