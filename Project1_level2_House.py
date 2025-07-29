# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Load Dataset
df = pd.read_csv("Housing.csv") 

# Step 3: View the actual column names
print("Columns in your dataset:\n", df.columns)

# Step 4: Clean and preprocess the data
df = df[['area', 'bedrooms', 'bathrooms', 'price']]

# Handle missing values 
df.dropna(inplace=True)

# Step 5: Define X (features) and y (target)
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 10: Visualize predictions vs actual values
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Housing Prices")
plt.show()
