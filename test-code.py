import pandas as pd

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
