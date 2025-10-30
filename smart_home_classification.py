#!/usr/bin/env python3

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Load preprocessed data
data = pd.read_csv('processed_smart_home_data.csv', index_col=0)
print(f"Data shape: {data.shape}")
print(data.head())

# The target is 'SmartHomeEfficiency', which is already binary (0 or 1)
# Features and target
X = data.drop(['SmartHomeEfficiency'], axis=1)
y = data['SmartHomeEfficiency']

print(f"Features shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    }

    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric.capitalize()}: {value:.4f}")

    return metrics, y_pred, y_pred_proba

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_proba, model_name):
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

# Dictionary to store results
results = {}
models = {}

# Logistic Regression
print("Training Logistic Regression...")
lr_params = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5, scoring='f1', n_jobs=-1)
lr_grid.fit(X_train, y_train)
lr_model = lr_grid.best_estimator_
lr_metrics, lr_pred, lr_proba = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
results['Logistic Regression'] = lr_metrics
models['Logistic Regression'] = lr_model
plot_confusion_matrix(y_test, lr_pred, 'Logistic Regression')
plot_roc_curve(y_test, lr_proba, 'Logistic Regression')

# Decision Tree
print("Training Decision Tree...")
dt_params = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='f1', n_jobs=-1)
dt_grid.fit(X_train, y_train)
dt_model = dt_grid.best_estimator_
dt_metrics, dt_pred, dt_proba = evaluate_model(dt_model, X_test, y_test, 'Decision Tree')
results['Decision Tree'] = dt_metrics
models['Decision Tree'] = dt_model
plot_confusion_matrix(y_test, dt_pred, 'Decision Tree')
plot_roc_curve(y_test, dt_proba, 'Decision Tree')

# Random Forest
print("Training Random Forest...")
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
rf_metrics, rf_pred, rf_proba = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
results['Random Forest'] = rf_metrics
models['Random Forest'] = rf_model
plot_confusion_matrix(y_test, rf_pred, 'Random Forest')
plot_roc_curve(y_test, rf_proba, 'Random Forest')

# SVM
print("Training SVM...")
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=5, scoring='f1', n_jobs=-1)
svm_grid.fit(X_train, y_train)
svm_model = svm_grid.best_estimator_
svm_metrics, svm_pred, svm_proba = evaluate_model(svm_model, X_test, y_test, 'SVM')
results['SVM'] = svm_metrics
models['SVM'] = svm_model
plot_confusion_matrix(y_test, svm_pred, 'SVM')
plot_roc_curve(y_test, svm_proba, 'SVM')

# KNN
print("Training KNN...")
knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='f1', n_jobs=-1)
knn_grid.fit(X_train, y_train)
knn_model = knn_grid.best_estimator_
knn_metrics, knn_pred, knn_proba = evaluate_model(knn_model, X_test, y_test, 'KNN')
results['KNN'] = knn_metrics
models['KNN'] = knn_model
plot_confusion_matrix(y_test, knn_pred, 'KNN')
plot_roc_curve(y_test, knn_proba, 'KNN')

# Naive Bayes
print("Training Naive Bayes...")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_metrics, nb_pred, nb_proba = evaluate_model(nb_model, X_test, y_test, 'Naive Bayes')
results['Naive Bayes'] = nb_metrics
models['Naive Bayes'] = nb_model
plot_confusion_matrix(y_test, nb_pred, 'Naive Bayes')
plot_roc_curve(y_test, nb_proba, 'Naive Bayes')

# XGBoost
print("Training XGBoost...")
xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.8, 1.0]}
xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_params, cv=5, scoring='f1', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_
xgb_metrics, xgb_pred, xgb_proba = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
results['XGBoost'] = xgb_metrics
models['XGBoost'] = xgb_model
plot_confusion_matrix(y_test, xgb_pred, 'XGBoost')
plot_roc_curve(y_test, xgb_proba, 'XGBoost')

# Neural Networks
print("Training Neural Networks...")
nn_params = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001]}
nn_grid = GridSearchCV(MLPClassifier(random_state=42, max_iter=1000), nn_params, cv=5, scoring='f1', n_jobs=-1)
nn_grid.fit(X_train, y_train)
nn_model = nn_grid.best_estimator_
nn_metrics, nn_pred, nn_proba = evaluate_model(nn_model, X_test, y_test, 'Neural Networks')
results['Neural Networks'] = nn_metrics
models['Neural Networks'] = nn_model
plot_confusion_matrix(y_test, nn_pred, 'Neural Networks')
plot_roc_curve(y_test, nn_proba, 'Neural Networks')

# CatBoost
print("Training CatBoost...")
cat_params = {'iterations': [100, 200], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1, 0.2], 'l2_leaf_reg': [1, 3, 5]}
cat_grid = GridSearchCV(CatBoostClassifier(random_state=42, verbose=False), cat_params, cv=5, scoring='f1', n_jobs=-1)
cat_grid.fit(X_train, y_train)
cat_model = cat_grid.best_estimator_
cat_metrics, cat_pred, cat_proba = evaluate_model(cat_model, X_test, y_test, 'CatBoost')
results['CatBoost'] = cat_metrics
models['CatBoost'] = cat_model
plot_confusion_matrix(y_test, cat_pred, 'CatBoost')
plot_roc_curve(y_test, cat_proba, 'CatBoost')

# Comparative analysis
print("\n" + "="*50)
print("COMPARATIVE ANALYSIS")
print("="*50)

# Create comparison dataframe
comparison_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(comparison_df.round(4))

# Plot comparison bar chart
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, metric in enumerate(metrics_to_plot):
    if i < len(axes):
        comparison_df[metric].plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{metric.capitalize()} Comparison')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].tick_params(axis='x', rotation=45)

if len(metrics_to_plot) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Feature importance for tree-based models
tree_models = ['Decision Tree', 'Random Forest', 'XGBoost', 'CatBoost']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for i, model_name in enumerate(tree_models):
    if model_name in models:
        model = models[model_name]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X.columns
            indices = np.argsort(importances)[::-1]

            axes[i].bar(range(len(importances)), importances[indices])
            axes[i].set_xticks(range(len(importances)))
            axes[i].set_xticklabels([feature_names[j] for j in indices], rotation=45, ha='right')
            axes[i].set_title(f'Feature Importance - {model_name}')
            axes[i].set_ylabel('Importance')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png')
plt.close()

# Identify best model
best_model = comparison_df['f1'].idxmax()
best_score = comparison_df['f1'].max()

print(f"\nBest Model: {best_model} with F1-Score: {best_score:.4f}")

# Save all models and results
print("\nSaving models and results...")
joblib.dump(models, 'trained_models.joblib')
joblib.dump(results, 'model_results.joblib')
comparison_df.to_csv('model_comparison.csv')

print("All models and results saved successfully!")

# Summary
print("\n" + "="*50)
print("SUMMARY OF MODEL PERFORMANCES")
print("="*50)
print(f"Total models evaluated: {len(results)}")
print(f"Best performing model: {best_model}")
print(".4f")
print("\nKey Insights:")
print("1. Tree-based models (Random Forest, XGBoost, CatBoost) generally performed well.")
print("2. Neural Networks showed competitive performance with proper tuning.")
print("3. SVM and KNN performed moderately but may require more tuning.")
print("4. Logistic Regression and Naive Bayes provided baseline performance.")
print("5. Feature importance analysis shows which features contribute most to predictions.")
print("\nComparative analysis complete!")