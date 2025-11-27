# Import libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print("Feature Names:", iris.feature_names)
print("Target Classes:", target_names)
print("\nSample Data (first 5 rows):")
print(X[:5])


# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
knn_pred = knn.predict(X_test)

# Train Decision Tree model
dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

# Predictions
dt_pred = dt.predict(X_test)

# KNN Evaluation
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))

# Decision Tree Evaluation
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12,5))

cm_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0])
axes[0].set_title("KNN Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

cm_dt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm_dt, annot=True, cmap='Greens', fmt='d', ax=axes[1])
axes[1].set_title("Decision Tree Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

print('\n✓ Saving confusion matrices to ml4_confusion_matrices.png')
fig.savefig('ml4_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close(fig)
