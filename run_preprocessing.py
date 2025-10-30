#!/usr/bin/env python3

# Install kagglehub if not already installed
import subprocess
subprocess.check_call(['pip', 'install', 'kagglehub', '-q'])

# Import necessary libraries
import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
import time

# Set random seed for reproducibility
np.random.seed(42)

# Download the dataset
try:
    path = kagglehub.dataset_download("rabieelkharoua/predict-smart-home-device-efficiency-dataset")
    print(f"Dataset successfully downloaded to: {path}")

    # List the files in the downloaded directory
    files = os.listdir(path)
    print(f"Files in the dataset: {files}")

    # Read the CSV file(s)
    if files:
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            file_path = os.path.join(path, csv_files[0])
            data = pd.read_csv(file_path, low_memory=False)
            print(f"Successfully read {csv_files[0]}")
            print(data.head())
        else:
            print("No CSV files found.")
    else:
        print("No files found.")

except Exception as e:
    print(f"Error: {e}")

# Data Cleaning
print("Starting data cleaning...")

# Rename columns to remove spaces and units
data.columns = [col[:-5].replace(' ', '_') if 'kW' in col else col for col in data.columns]

# Drop rows with NaN values
data = data.dropna()

# Remove duplicate rows
data = data.drop_duplicates()

# Drop redundant columns
if 'House_overall' in data.columns:
    data.drop(['House_overall'], axis=1, inplace=True)
if 'Solar' in data.columns:
    data.drop(['Solar'], axis=1, inplace=True)

# Handle cloudCover: drop non-numeric and convert (if exists)
if 'cloudCover' in data.columns:
    data = data[data['cloudCover'] != 'cloudCover']
    data['cloudCover'] = pd.to_numeric(data['cloudCover'], errors='coerce')
    data = data.dropna(subset=['cloudCover'])

# Convert time column to datetime index (if exists)
if 'time' in data.columns:
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(data['time'].iloc[0])))
    time_index = pd.date_range(start_time, periods=len(data), freq='min')
    data = data.set_index(time_index)
    data = data.drop(['time'], axis=1)

print(f"Data shape after cleaning: {data.shape}")
print(data.dtypes)

# Feature Engineering
print("Performing feature engineering...")

# Create aggregated features
if all(col in data.columns for col in ['Kitchen_12', 'Kitchen_14', 'Kitchen_38']):
    data['kitchen'] = data['Kitchen_12'] + data['Kitchen_14'] + data['Kitchen_38']
if all(col in data.columns for col in ['Furnace_1', 'Furnace_2']):
    data['Furnace'] = data['Furnace_1'] + data['Furnace_2']

# Time-based features (if datetime index)
if isinstance(data.index, pd.DatetimeIndex):
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# Weather categories (if weather columns exist)
if 'temperature' in data.columns:
    data['temp_category'] = pd.cut(data['temperature'], bins=[-np.inf, 0, 15, 25, np.inf], labels=['freezing', 'cold', 'mild', 'hot'])
if 'humidity' in data.columns:
    data['humidity_category'] = pd.cut(data['humidity'], bins=[0, 30, 60, 90, 100], labels=['dry', 'comfortable', 'humid', 'very_humid'])

# Efficiency metric (if gen and use exist)
if 'gen' in data.columns and 'use' in data.columns:
    data['efficiency'] = data['gen'] / (data['use'] + 1e-6)  # Avoid division by zero

print(f"Data shape after feature engineering: {data.shape}")
print(data.head())

# EDA: Distributions
print("EDA: Distributions")

# Numerical columns
num_cols = data.select_dtypes(include=[np.number]).columns
for col in num_cols[:5]:  # Plot first 5 to avoid too many
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'{col}_dist.png')
    plt.close()  # Close to avoid display issues

# Categorical columns
cat_cols = data.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    data[col].value_counts().plot(kind='bar')
    plt.title(f'Bar plot of {col}')
    plt.savefig(f'{col}_bar.png')
    plt.close()

# EDA: Correlations
print("EDA: Correlations")

# Correlation heatmap for numerical features
plt.figure(figsize=(12, 8))
corr = data[num_cols].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# Top correlations with target (assuming 'use' or 'efficiency')
target = 'efficiency' if 'efficiency' in data.columns else 'use'
if target in num_cols:
    correlations = data[num_cols].corr()[target].abs().sort_values(ascending=False)
    print(f"Top correlations with {target}:")
    print(correlations.head(10))
else:
    print(f"Target {target} not in numerical columns")

# EDA: Time-series plots
print("EDA: Time-series plots")

# Daily means for key features (if datetime index)
key_features = ['use', 'gen', 'temperature', 'humidity']
existing_features = [f for f in key_features if f in data.columns]
if isinstance(data.index, pd.DatetimeIndex):
    for col in existing_features:
        plt.figure(figsize=(15, 5))
        data[col].resample('D').mean().plot()
        plt.title(f'Daily mean of {col}')
        plt.savefig(f'{col}_daily.png')
        plt.close()

    # Monthly energy consumption
    if 'use' in data.columns:
        plt.figure(figsize=(15, 5))
        data['use'].resample('M').sum().plot()
        plt.title('Monthly total energy use')
        plt.savefig('monthly_use.png')
        plt.close()
else:
    print("No datetime index available for time-series plots")

# Identify features and target
print("Identifying features and target")

# Target variable
target_col = 'SmartHomeEfficiency'  # Based on the data, this is the target
print(f"Target column: {target_col}")

# Features
if target_col in data.columns:
    X = data.drop(target_col, axis=1)
    y = data[target_col]
else:
    X = data
    y = None
    print(f"Warning: Target column '{target_col}' not found")

# Separate numerical and categorical
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical features: {numerical_cols}")
print(f"Categorical features: {categorical_cols}")
print(f"Data shapes: X={X.shape}, y={y.shape if y is not None else 'None'}")

# Create preprocessing pipeline
print("Creating preprocessing pipeline")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Fit preprocessor
X_processed = preprocessor.fit_transform(X)
print(f"Processed data shape: {X_processed.shape}")

# Save processed data and objects
print("Saving processed data and preprocessing objects")

# Save processed data
processed_data = pd.DataFrame(X_processed, index=X.index)
processed_data[target_col] = y.values
processed_data.to_csv('processed_smart_home_data.csv', index=True)
print("Processed data saved as 'processed_smart_home_data.csv'")

# Save preprocessor
dump(preprocessor, 'smart_home_preprocessor.joblib')
print("Preprocessor saved as 'smart_home_preprocessor.joblib'")

# Save raw cleaned data
data.to_csv('cleaned_smart_home_data.csv', index=True)
print("Cleaned data saved as 'cleaned_smart_home_data.csv'")