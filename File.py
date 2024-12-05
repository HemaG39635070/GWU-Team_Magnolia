#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# %%
df = pd.read_csv("Airline_data.csv")

print(df.head())
print(df.info())
# %%
df.info()

# %%
df.columns

# %%
df.describe()

# %%
df.shape

# %%
df=df.drop(['id','Unnamed: 0'],axis=1)

# %%
df.describe(include = object)

# %%
df.isnull().sum()

# %%
df.dropna(subset=['Arrival Delay in Minutes'], inplace=True)

# %%
df.isnull().any()

# %%
df.duplicated().any()

# %%
numeric_columns = df.select_dtypes(include=['number']).columns
plt.figure(figsize=(5, 3))

for col in numeric_columns:
    sns.boxplot(data=df, y=col)
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
    plt.show()
# %%

def Outliers(df,col, output_file="df_clean.csv"):
    Q1 = df[col].quantile(q=0.25)
    Q3 = df[col].quantile(q=0.75)
    IQR = df[col].apply(stats.iqr)
    df_clean = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).any(axis=1)]
    print("Number of rows before removing outliers:", len(df))
    print("Number of rows after removing outliers:", len(df_clean))
    df_clean.to_csv(output_file, index=False)
    print(f"Cleaned DataFrame saved as '{output_file}'.")

    return df_clean


columns = ['Flight Distance']
df_cleaned = Outliers(df, columns, output_file="cleaned_Airport_data.csv")

print(df_cleaned.head())

# %%
df_cleaned.info()

# %%
columns_to_encode = ["satisfaction","Gender","Customer Type","Type of Travel","Class"]

for col in columns_to_encode:
    le = preprocessing.LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    l = list(le.classes_)
df_cleaned.head()

#Gender --> Female - 0, Male - 1
#Customer Type --> Loyal - 0, Disloyal - 1
#Type of Travel --> Business Travel - 0, Personal Travel - 1
#Class --> Business - 0,  Economy - 1, Economy Plus - 2
#Satisfaction --> Non Satisfied - 0, Satisfied - 1

# %%

