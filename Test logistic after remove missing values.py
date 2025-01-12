# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import data file from GItHub
url = 'https://raw.githubusercontent.com/khoocheeshin/HIA303BreastCancerProject/refs/heads/main/Selected-data-and-description-files/breast-cancer-wisconsin.data'

# Load the data file into a DataFrame
bcw = pd.read_csv(url, header=None)

# DataFrame Structuring
# Rename columns
column_names = [
    'ID', 'clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape',
    'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
    'bland_chromatin', 'normal_nucleoli', 'mitoses', 'diagnosis'
]
bcw.columns = column_names

# Display the first 20 rows to confirm it's loaded correctly
bcw.head(20)

# Overview of DataFrame: range of rows, number of columns, column names, non null counts and data types
bcw.info()

# Data Cleaning
# 1. Check for duplicate rows
duplicates = bcw[bcw.duplicated()]

# Print information about duplicates
if duplicates.empty:
    print("No duplicate rows found in the dataset.")
else:
    print(f"Found {len(duplicates)} duplicate rows in the dataset.")
    print("Duplicate rows:")
    print(duplicates)

# Remove duplicate rows
bcw_cleaned = bcw.drop_duplicates()

# Verify the duplicates are removed
print(f"\nDataset after removing duplicates has {bcw_cleaned.shape[0]} rows.")

# 2. Check for missing values
# Replace the missing values (i.e. with “?”) with a NaN (Not a Number) value 
bcw_cleaned = bcw_cleaned.replace('?', np.nan)

# Count the total number of NaN values in each column
nan_counts = bcw_cleaned.isnull().sum()
# Display the total NaN counts in each column
print("Total NaN counts in each column:\n", nan_counts)

# Isolate rows with missing values
rows_with_missing_values = bcw_cleaned[bcw_cleaned.isnull().any(axis=1)]

# Display rows with missing values
print("Rows with missing values:")
print(rows_with_missing_values)

# Count the number of rows with missing values
print(f"\nTotal number of rows with missing values: {len(rows_with_missing_values)}")

# Check the distribution of the diagnosis column of bcw_cleaned 
print("Distribution of 'diagnosis' column before removing rows with missing values:")
print(bcw_cleaned['diagnosis'].value_counts(normalize=True) * 100)

# Remove rows with missing values
bcw_cleaned = bcw_cleaned.dropna()

# Verify the rows with missing values are removed
nan_counts_post = bcw_cleaned.isnull().sum()
print("Total NaN counts in each column after removal:\n", nan_counts_post)
print(bcw_cleaned.info())

# Check the distribution of the diagnosis column after removal
print("\nDistribution of 'diagnosis' column after removing rows with missing values:")
print(bcw_cleaned['diagnosis'].value_counts(normalize=True) * 100)

# 3. Check for data consistency in the target column (2 = benign and 4 = malignant) 
# and feature columns (integer 1 to 10)
for column in bcw_cleaned.columns:
    if column != 'ID':
        unique_values = bcw_cleaned[column].unique()
        print(f"Unique values in {column}: {unique_values}")
        
# Data Standardization
# 1. Convert datatype of bare_nuclei column from object to integer
bcw_cleaned['bare_nuclei'] = bcw_cleaned['bare_nuclei'].astype(int)

# 2. Standardization for data in target column
# Map the diagnosis values: 2 -> 0 (benign), 4 -> 1 (malignant)
bcw_cleaned['diagnosis'] = bcw_cleaned['diagnosis'].map({2: 0, 4: 1})
bcw_cleaned.info()
# Verify the mapping
print("Updated unique values in 'diagnosis':", bcw_cleaned['diagnosis'].unique())

# 3. Feature scaling using RobustScaler
features = [
    'clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape',
    'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
    'bland_chromatin', 'normal_nucleoli', 'mitoses'
]

# Initialize the scaler
scaler = RobustScaler()

# Apply scaling to the features (exclude ID and diagnosis columns)
X_scaled = scaler.fit_transform(bcw_cleaned[features])

# Create a new DataFrame with scaled features and diagnosis
bcw_scaled = pd.DataFrame(X_scaled, columns=features)
# Reset index of bcw_cleaned to prevent misalignment
bcw_cleaned.reset_index(drop=True, inplace=True)
bcw_scaled['diagnosis'] = bcw_cleaned['diagnosis']

# Display the first few rows of the scaled DataFrame
bcw_scaled.head()

# Class Balancing
# SMOTE (Synthetic Minority Over-sampling Technique)
# Separate features and target variable (diagnosis)
X = bcw_scaled.drop(columns=['diagnosis'])
y = bcw_scaled['diagnosis']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)

# Fit and apply SMOTE on the dataset
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a new DataFrame with balanced features and target variable
bcw_balanced = pd.concat([X_resampled, y_resampled], axis=1)

# Display the class distribution before and after oversampling
print("Class distribution before oversampling:\n", y.value_counts())
print("\nClass distribution after oversampling:\n", y_resampled.value_counts())

# Data Splitting
# Perform train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    bcw_balanced.drop(columns=['diagnosis']),
    bcw_balanced['diagnosis'],
    test_size=0.2, random_state=42)

# Create DataFrames for training and testing sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Display the first 20 rows of train data
train_data.head(20)

# Feature Selection
# Feature selection using Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Extract features and target variable
X_rf = bcw_balanced.drop(columns=['diagnosis'])
y_rf = bcw_balanced['diagnosis']

# Initialize RandomForest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)

# 5-Fold Cross-Validation using AUC as the scoring metric
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(rf, X_rf, y_rf, cv=cv, scoring='roc_auc')

# Fit the Random Forest model to get feature importances
rf.fit(X_rf, y_rf)

# Get feature importances
feature_importances = rf.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X_rf.columns,
    'Importance': feature_importances
})

print("Cross-Validation AUC Scores:", scores)
print("Average AUC Score:", np.mean(scores))
print("\nFeature Importances:\n", feature_importance_df)

# Sort the DataFrame by importance in ascending order for the plot
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)

plt.title('Feature Importance Ranked by Gini Impurity')
plt.xlabel('Gini Impurity Decrease')
plt.ylabel('Features')
plt.show()


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Define features based on importance
all_features = X_train.columns.tolist()
top_8_features = ['uniformity_cell_size', 'uniformity_cell_shape', 
                  'single_epithelial_cell_size', 'bare_nuclei',  
                  'bland_chromatin', 'clump_thickness', 'normal_nucleoli',
                  'marginal_adhesion']
top_7_features = ['uniformity_cell_size', 'uniformity_cell_shape', 
                  'single_epithelial_cell_size', 'bare_nuclei',  
                  'bland_chromatin', 'clump_thickness', 'normal_nucleoli']

# Hyperparameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],  # L1 = Lasso, L2 = Ridge
    'solver': ['liblinear']  # Supports both l1 and l2
}

# Function to train, tune, and evaluate a model
def train_and_evaluate(features, feature_set_name):
    print(f"\nTraining Logistic Regression with {feature_set_name}...")
    X_train_subset = X_train[features]
    X_test_subset = X_test[features]
    
    # Initialize logistic regression and perform GridSearchCV
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(
        logreg, 
        param_grid, 
        cv=5,  # Perform 5-fold cross-validation on training data
        scoring='roc_auc', 
        n_jobs=-1, 
        verbose=1
    )
    grid_search.fit(X_train_subset, y_train)
    
    # Best hyperparameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"\nBest Parameters for {feature_set_name}: {best_params} with AUC = {best_score:.4f} ")
    
    # Retrain the model on the entire training set using the best parameters
    optimized_LR = LogisticRegression(
        random_state=42,
        max_iter=1000,
        **best_params
    )
    optimized_LR.fit(X_train_subset, y_train)
    
    # Evaluate the final model on the test set
    y_pred = optimized_LR.predict(X_test_subset)
    y_pred_proba = optimized_LR.predict_proba(X_test_subset)[:, 1]
    
    print(f"\nClassification Report for testing LR with {feature_set_name}:\n")
    print(classification_report(y_test, y_pred, digits=4))
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score for LR with {feature_set_name}: {auc_score:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()  # Extract true negatives, false positives, false negatives, true positives
    
    # Visualization of Confusion Matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # Sensitivity 
    sensitivity = tp / (tp + fn)
    # Specificity
    specificity = tn / (tn + fp)
    # Positive Predictive Value (PPV)
    ppv = tp / (tp + fp)
    # Negative Predictive Value (NPV)
    npv = tn / (tn + fn)
    
    print(f"Sensitivity for {feature_set_name}: {sensitivity:.4f}")
    print(f"Specificity for {feature_set_name}: {specificity:.4f}")
    print(f"Positive Predictive Value (PPV) for {feature_set_name}: {ppv:.4f}")
    print(f"Negative Predictive Value (NPV) for {feature_set_name}: {npv:.4f}")
    
    
    return best_params, auc_score, sensitivity, specificity, ppv, npv

# Train and evaluate models
results_LR = {}
results_LR['All Features'] = train_and_evaluate(all_features, "All Features")
results_LR['Top 8 Features'] = train_and_evaluate(top_8_features, "Top 8 Features")
results_LR['Top 7 Features'] = train_and_evaluate(top_7_features, "Top 7 Features")




