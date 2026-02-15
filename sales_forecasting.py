import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create dummy historical data (In a real project, use pd.read_csv('sales_data.csv'))
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marketing_Spend_USD': [1000, 1200, 1500, 1100, 1300, 1600, 1400, 1700, 1250, 1800],
    'Discount_Percentage': [5, 10, 15, 5, 10, 20, 10, 25, 5, 20],
    'Website_Visitors': [5000, 5500, 6000, 5200, 5800, 6500, 6100, 7000, 5400, 7200],
    'Total_Sales_USD': [15000, 16500, 19000, 15500, 17500, 21000, 18500, 23000, 16000, 24000] # Target Variable
}

df = pd.DataFrame(data)

# 2. Define Features (X) and Target (y)
X = df[['Marketing_Spend_USD', 'Discount_Percentage', 'Website_Visitors']]
y = df['Total_Sales_USD']

# 3. Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Multiple Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Predictions on the Test Data
y_pred = model.predict(X_test)

# 6. Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R-squared Score: {r2:.2f} (Closer to 1.0 is better)\n")

# 7. Forecast next month's sales based on new inputs
next_month_data = pd.DataFrame({
    'Marketing_Spend_USD': [2000], 
    'Discount_Percentage': [15], 
    'Website_Visitors': [7500]
})

forecast = model.predict(next_month_data)
print("--- Next Month Forecast ---")
print(f"Predicted Sales: ${forecast[0]:.2f}")