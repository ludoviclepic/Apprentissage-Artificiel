import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
X = np.load('X_preprocessed.npy')
y = np.load('y.npy')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best hyperparameters for Random Forest
rf_best_params = {
    'n_estimators': 400,
    'max_depth': None,
    'min_samples_split': 5,
    'min_samples_leaf': 1
}

# Best hyperparameters for SVM
svm_best_params = {
    'C': 100,
    'gamma': 1,
    'kernel': 'rbf'
}

# Initialize models with best hyperparameters
rf_best = RandomForestClassifier(**rf_best_params, random_state=42)
svm_best = SVC(**svm_best_params, random_state=42)

# Train models
rf_best.fit(X_train, y_train)
svm_best.fit(X_train, y_train)

# Evaluate models
rf_predictions = rf_best.predict(X_test)
svm_predictions = svm_best.predict(X_test)

# Print results
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))
