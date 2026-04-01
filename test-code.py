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