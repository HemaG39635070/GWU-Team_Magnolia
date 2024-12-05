#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# %%
df = pd.read_csv("C:/Users/g/OneDrive/Desktop/GWU/202408/DATS 6103 Intro to Data Mining/Mod 3/Project/Airline_data.csv")

# %%
df=df.drop(['id','Unnamed: 0'],axis=1)
df.dropna(subset=['Arrival Delay in Minutes'], inplace=True)

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
df_cleaned.head()
# %%


# Service attributes to plot
service_attributes = [
    'Inflight wifi service',
    'Departure/Arrival time convenient',
    'Ease of Online booking',
    'Gate location',
    'Food and drink',
    'Online boarding',
    'Seat comfort',
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Inflight service',
    'Cleanliness'
]
# Create two separate figures for satisfied and non-satisfied customers
def plot_satisfaction_vertical_boxplots(df):
    # Satisfied customers (1)
    plt.figure(figsize=(20, 10))
    satisfied_df = df[df['satisfaction'] == 1]
    
    for i, attr in enumerate(service_attributes, 1):
        plt.subplot(4, 4, i)
        sns.boxplot(y=satisfied_df[attr])
        plt.title(f'Satisfied Customers: {attr}', fontsize=10)
        plt.ylabel('Rating', fontsize=8)
        plt.xticks([])  # Remove x-ticks for cleaner look
    
    plt.tight_layout()
    plt.suptitle('Service Attributes for Satisfied Customers', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Non-satisfied customers (0)
    plt.figure(figsize=(20, 10))
    non_satisfied_df = df[df['satisfaction'] == 0]
    
    for i, attr in enumerate(service_attributes, 1):
        plt.subplot(4, 4, i)
        sns.boxplot(y=non_satisfied_df[attr])
        plt.title(f'Non-Satisfied Customers: {attr}', fontsize=10)
        plt.ylabel('Rating', fontsize=8)
        plt.xticks([])  # Remove x-ticks for cleaner look
    
    plt.tight_layout()
    plt.suptitle('Service Attributes for Non-Satisfied Customers', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

# Call the function with your cleaned dataframe
plot_satisfaction_vertical_boxplots(df_cleaned)
# %%

# Create two separate figures for satisfied and non-satisfied customers
def plot_satisfaction_violin_plots(df):
    # Satisfied customers (1)
    plt.figure(figsize=(20, 10))
    satisfied_df = df[df['satisfaction'] == 1]
    
    for i, attr in enumerate(service_attributes, 1):
        plt.subplot(4, 4, i)
        sns.violinplot(y=satisfied_df[attr])
        plt.title(f'Satisfied Customers: {attr}', fontsize=10)
        plt.ylabel('Rating', fontsize=8)
        plt.xticks([])  # Remove x-ticks for cleaner look
    
    plt.tight_layout()
    plt.suptitle('Service Attributes for Satisfied Customers', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Non-satisfied customers (0)
    plt.figure(figsize=(20, 10))
    non_satisfied_df = df[df['satisfaction'] == 0]
    
    for i, attr in enumerate(service_attributes, 1):
        plt.subplot(4, 4, i)
        sns.violinplot(y=non_satisfied_df[attr])
        plt.title(f'Non-Satisfied Customers: {attr}', fontsize=10)
        plt.ylabel('Rating', fontsize=8)
        plt.xticks([])  # Remove x-ticks for cleaner look
    
    plt.tight_layout()
    plt.suptitle('Service Attributes for Non-Satisfied Customers', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

# Call the function with your cleaned dataframe
plot_satisfaction_violin_plots(df_cleaned)
# %%
