import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("wine.csv")

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Split data into features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred, zero_division=0))

# Model 2: SGD Classifier
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)
print("SGD Classifier")
print("Accuracy:", accuracy_score(y_test, sgd_pred))
print(classification_report(y_test, sgd_pred, zero_division=0))

# Model 3: Support Vector Classifier
svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print("SVC")
print("Accuracy:", accuracy_score(y_test, svc_pred))
print(classification_report(y_test, svc_pred, zero_division=0))

# Visualization

# Heatmap of feature correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Scatter plot: Actual vs Predicted (Random Forest)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=rf_pred, alpha=0.6)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality (RF)")
plt.title("Actual vs Predicted Wine Quality (Random Forest)")
plt.show()
