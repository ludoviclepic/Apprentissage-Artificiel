import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression  # added for lasso regularization

# Load preprocessed data
X = np.load('X_preprocessed.npy')
y = np.load('y.npy')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
rf = RandomForestClassifier(random_state=42)
svm = SVC(kernel='rbf', random_state=42)

# Define expanded hyperparameter grids
rf_param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['rbf', 'linear']
}

# Added Lasso regularized logistic regression grid search (L1 penalty)
logreg = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
logreg_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]
}

# Perform grid search
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy')
svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy')
logreg_grid_search = GridSearchCV(logreg, logreg_param_grid, cv=5, scoring='accuracy')

# Train models
rf_grid_search.fit(X_train, y_train)
svm_grid_search.fit(X_train, y_train)
logreg_grid_search.fit(X_train, y_train)

# Evaluate models
rf_best = rf_grid_search.best_estimator_
svm_best = svm_grid_search.best_estimator_
logreg_best = logreg_grid_search.best_estimator_

rf_predictions = rf_best.predict(X_test)
svm_predictions = svm_best.predict(X_test)
logreg_predictions = logreg_best.predict(X_test)

# Print results
print("Random Forest Best Parameters:", rf_grid_search.best_params_)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

print("SVM Best Parameters:", svm_grid_search.best_params_)
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

print("Lasso Logistic Regression Best Parameters:", logreg_grid_search.best_params_)
print("Lasso Logistic Regression Accuracy:", accuracy_score(y_test, logreg_predictions))
print("Lasso Logistic Regression Classification Report:\n", classification_report(y_test, logreg_predictions))
