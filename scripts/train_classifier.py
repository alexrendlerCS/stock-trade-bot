import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
FEATURE_CSV = 'combined_features.csv'  # Use the combined features file

# Load data
print(f"Loading data from {FEATURE_CSV}...")
df = pd.read_csv(FEATURE_CSV, index_col=0)

# Print columns of loaded DataFrame for verification
print("\nColumns in loaded DataFrame:", df.columns)

# Print unique tickers
print("\nUnique tickers in dataset:", df['ticker'].unique())

# Drop rows with any NaN values
df = df.dropna()

# Debug: Check feature/target alignment
print("\nSample of features and targets:")
print(df[['ticker', 'Close', 'next_return', 'target_3d', 'target_5d']].tail(10))

# Choose which target to use: 'target_3d' or 'target_5d'
targets_to_test = ['target_3d', 'target_5d']

for target_col in targets_to_test:
    print(f"\n=== Training for {target_col} ===")
    # Drop rows with NaN in the target
    df_target = df.dropna(subset=[target_col])
    print(f"Class distribution for {target_col}:\n", df_target[target_col].value_counts())
    
    # Features and target
    feature_cols = [col for col in df_target.columns if col not in ['target_3d', 'target_5d', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'next_return', 'return_3d', 'return_5d', 'ticker']]
    X = df_target[feature_cols]
    y = df_target[target_col].astype(int)
    
    # Debug: Check for constant features
    print("\nFeature variances:")
    print(X.var())
    
    # Debug: Check for duplicated data
    print("\nUnique rows in X:", X.drop_duplicates().shape[0], "/", X.shape[0])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Debug: Check for data leakage (index overlap)
    print("\nTrain indices:", X_train.index[:5], "...", X_train.index[-5:])
    print("Test indices:", X_test.index[:5], "...", X_test.index[-5:])
    print("Overlap in train/test indices:", set(X_train.index) & set(X_test.index))
    
    # Train Random Forest with class_weight balanced
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Feature importance analysis
    importances = clf.feature_importances_
    feature_names = X.columns
    sorted_idx = importances.argsort()[::-1]
    
    print("\nTop 10 most important features:")
    for idx in sorted_idx[:10]:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances[sorted_idx[:20]], y=feature_names[sorted_idx[:20]])
    plt.title(f'Top 20 Feature Importance for {target_col}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{target_col}.png')
    plt.close()
    
    # Save model for each target
    model_file = f'rf_model_{target_col}.pkl'
    joblib.dump(clf, model_file)
    print(f"Model saved to {model_file}")
    
    # Save feature importance to CSV
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(f'feature_importance_{target_col}.csv')
    print(f"Feature importance saved to feature_importance_{target_col}.csv") 