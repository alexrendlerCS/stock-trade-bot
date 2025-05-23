import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Parameters
FEATURE_CSV = 'AAPL_features.csv'  # Change if you use a different ticker
MODEL_FILE = 'rf_model.pkl'

# Load data
df = pd.read_csv(FEATURE_CSV, index_col=0)

# Drop rows with any NaN values
df = df.dropna()

# Print class distribution
print("Class distribution:\n", df['target'].value_counts())

# Debug: Check feature/target alignment
print("\nSample of features and target:")
print(df[['Close', 'next_return', 'target']].tail(10))

# Features and target
feature_cols = [col for col in df.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'next_return']]
X = df[feature_cols]
y = df['target']

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

# Save model
joblib.dump(clf, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}") 