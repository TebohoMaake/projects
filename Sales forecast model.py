# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Teboho Edward Maake
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,  r2_score

df = pd.read_csv("C:/Projects/Walmart.csv")  

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Day'] = df['Date'].dt.day
df = df.sort_values(['Store', 'Date'])
df['Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
df['Lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
df = df.dropna()
df.drop(columns=['Date'], inplace=True)
    
features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 
           'Unemployment', 'Year', 'Month', 'Week', 'Day', 'Lag_1', 'Lag_2']
target = 'Weekly_Sales'

X = df[features]
y = df[target]
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xticks(rotation=45)
plt.title('Feature Importance in Sales Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Weekly Sales')
plt.tight_layout()
plt.show()

# Function to forecast future sales
def forecast_sales(model, store, date, holiday_flag, temperature, 
                  fuel_price, cpi, unemployment, previous_sales):
    # Create input dataframe for prediction
    future_date = pd.to_datetime(date)
    input_data = pd.DataFrame({
        'Store': [store],
        'Holiday_Flag': [holiday_flag],
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'CPI': [cpi],
        'Unemployment': [unemployment],
        'Year': [future_date.year],
        'Month': [future_date.month],
        'Week': [future_date.isocalendar().week],
        'Day': [future_date.day],
        'Lag_1': [previous_sales[0]],
        'Lag_2': [previous_sales[1]]
    })
    
    prediction = model.predict(input_data)[0]
    return prediction

# My new model to predict sales
store_id = 1
forecast_date = '2025-03-17'
example_forecast = forecast_sales(
    rf_model,
    store_id,
    forecast_date,
    holiday_flag=0,
    temperature=60.0,
    fuel_price=3.5,
    cpi=220.0,
    unemployment=7.0,
    previous_sales=[df[df['Store'] == store_id]['Weekly_Sales'].iloc[-1], 
                   df[df['Store'] == store_id]['Weekly_Sales'].iloc[-2]]
)

print(f"\nForecast for Store {store_id} on {forecast_date}: ${example_forecast:,.2f}")
#########################################################################################################

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()






