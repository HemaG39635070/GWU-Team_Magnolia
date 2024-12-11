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
#EDA: Type of travellers, class and Satisfaction rates.

columns_to_plot = ['Type of Travel', 'Class', 'satisfaction']

for column in columns_to_plot:
    plt.figure(figsize=(6, 6))
    value_counts = df_cleaned[column].value_counts()
    value_counts.plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=plt.cm.Paired.colors,
        startangle=90
    )
    plt.title(f'Distribution of {column}')
    plt.ylabel('')
    plt.show()

#%%
# EDA : Analysing the satisfaction and dissatisfaction rate for different Age groups.

bins = [0, 18, 30, 40, 50, 60, 100]  
labels = ['<18', '18-30', '31-40', '41-50', '51-60', '60+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Age Group', hue='satisfaction', palette='coolwarm')
plt.title('Satisfaction Rates by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Passengers')
plt.show()

# Observation : Passengers between the ages of 18 and 30 are the most dissatisfied, while those between the ages of 41 and 50 are the most satisfied.

#%%
# Analysing the satisfaction between the loyal and disloyal customers.

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Customer Type', hue='satisfaction', palette='coolwarm')
plt.title('Satisfaction Rates by Customer Loyalty')
plt.xlabel('Customer Type')
plt.ylabel('Number of Passengers')
plt.show()

#Observation : It is evident that the percentage of dissatisfied customers is considerable for both loyal and disloyal customers. This raises the question of whether loyalty influences satisfaction.

#%%
# Statistical testing on loyalty and Age group

from scipy.stats import chi2_contingency

contingency_table_age = pd.crosstab(df['Age'], df['satisfaction'])
chi2, p, dof, expected = chi2_contingency(contingency_table_age)
print(f"Chi-square test p-value: {p}")

contingency_table_loyalty = pd.crosstab(df['Customer Type'], df['satisfaction'])
chi2, p, dof, expected = chi2_contingency(contingency_table_loyalty)
print(f"Chi-square test p-value: {p}")

# Observation :The p-value of 0 indicates that these variables are statistically significant .git

#%%
# Continous variable distribution to analyse if the predictors are heavily skewed 

continuous_variables = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']  

plt.figure(figsize=(12, 8))
for i, var in enumerate(continuous_variables, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[var], kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#%%
# Target Variable distribution

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='satisfaction', palette='coolwarm')
plt.title('Satisfaction Distribution')
plt.xlabel('Distribution')
plt.ylabel('Frequency')
plt.show()

#%%
# Log transformation for positively skewed data

df['Flight Distance'] = df['Flight Distance'].apply(lambda x: np.log(x + 1))
df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].apply(lambda x: np.log(x + 1))
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].apply(lambda x: np.log(x + 1))

sns.histplot(df['Departure Delay in Minutes'], kde=True)
plt.show()

#%%
# Correlation Matrix

numeric_columns = df.select_dtypes(include=['number'])
corr_matrix = numeric_columns.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

#%%
# Variance Inflation Factor

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

numeric_columns = df.select_dtypes(include=['number'])

X = add_constant(numeric_columns)

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

#%%
# Drop highly correlated variable

to_drop = set()

threshold = 0.8  
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]  
            to_drop.add(colname)

df = df.drop(columns=to_drop)

print("Dropped features due to high correlation:", to_drop)

#%%
# Train-Test split 

from sklearn.model_selection import train_test_split


X = df_01.drop('satisfaction', axis=1)  
y = df['satisfaction']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

#%%
# Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)


y_pred_lr = lr_model.predict(X_test)


print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)}")


#%%
# confusion matrix 

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)

disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

#%%
# ROC AUC Curve

from sklearn.metrics import roc_curve, roc_auc_score

y_pred_prob = lr_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

auc_score = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

#%%
# Train and Test prediction 

train_preds = lr_model.predict(X_train)
test_preds = lr_model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

#%%
# Cross Validation

from sklearn.model_selection import cross_val_score

cross_val_scores = cross_val_score(lr_model, X, y, cv=5)

print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean cross-validation score: {cross_val_scores.mean()}")

#%%
# Random Forrest Classification

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
