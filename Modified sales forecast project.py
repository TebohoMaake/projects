# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 23:02:26 2025

@author: Rembu
"""
#Teboho Maake
#Sales forecast
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from io import StringIO


df = pd.read_csv(C:/Projects/Walmart.csv/)

# Preprocessing and feature engineering
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Holiday_Month'] = df['Holiday_Flag'] * df['Month']
    df = df.sort_values(['Store', 'Date'])
    df['Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
    df['Lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
    return df.dropna()

# Remove influential points using Cook's Distance
def remove_influential_points(df, features, target):
    X = sm.add_constant(df[features])
    y = df[target]
    model = sm.OLS(y, X).fit()
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(df)
    mask = cooks_d < threshold
    print(f"Removed {len(df) - mask.sum()} influential points (Cook's Distance > {threshold:.4f})")
    return df[mask]

# Build and evaluate forecast model
def build_forecast_model(df, features, target):
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Baseline model: Mean sales per store
    baseline_preds = y_test.groupby(df.loc[X_test.index, 'Store']).transform('mean')
    
    # Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # A/B Testing Simulation
    np.random.seed(42)
    ab_mask = np.random.rand(len(y_test)) < 0.5  # 50% control, 50% treatment
    control_preds = baseline_preds[ab_mask]
    treatment_preds = y_pred[~ab_mask]
    control_actual = y_test[ab_mask]
    treatment_actual = y_test[~ab_mask]
    
    baseline_mae = mean_absolute_error(control_actual, control_preds)
    rf_mae = mean_absolute_error(treatment_actual, treatment_preds)
    t_stat, p_value = stats.ttest_ind(abs(control_actual - control_preds), 
                                    abs(treatment_actual - treatment_preds))
    
    print("\nA/B Testing Results:")
    print(f"Baseline MAE (Control Group): ${baseline_mae:,.2f}")
    print(f"Random Forest MAE (Treatment Group): ${rf_mae:,.2f}")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("The improvement is statistically significant (p < 0.05).")
    else:
        print("The improvement is not statistically significant (p >= 0.05).")
    
    # Model Performance Metrics (overall)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance Metrics (Overall):")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance in Sales Prediction')
    plt.tight_layout()
    plt.show()
    
    # Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Weekly Sales')
    plt.tight_layout()
    plt.show()
    
    return rf_model, y_test, y_pred

# Cost-benefit analysis
def cost_benefit_analysis(df, y_test, y_pred, baseline_preds):
    # Assumptions
    total_sales = df['Weekly_Sales'].sum()
    overstock_cost_rate = 0.20  # 20% loss on unsold inventory
    understock_cost_rate = 0.10  # 10% lost profit on missed sales
    development_cost = 5000  # Hypothetical one-time development cost ($)
    maintenance_cost = 1000  # Hypothetical annual maintenance cost ($)
    
    # Baseline error
    baseline_error = abs(y_test - baseline_preds)
    rf_error = abs(y_test - y_pred)
    error_reduction = baseline_error.mean() - rf_error.mean()
    
    # Estimate overstock/understock savings
    total_error_cost_baseline = baseline_error.mean() * (overstock_cost_rate + understock_cost_rate)
    total_error_cost_rf = rf_error.mean() * (overstock_cost_rate + understock_cost_rate)
    annual_savings = (total_error_cost_baseline - total_error_cost_rf) * len(y_test) * 52  # Scaled to yearly
    
    # Total costs
    total_cost = development_cost + maintenance_cost
    
    # Net benefit
    net_benefit = annual_savings - total_cost
    
    print("\nCost-Benefit Analysis:")
    print(f"Total Historical Sales: ${total_sales:,.2f}")
    print(f"Average Error Reduction: ${error_reduction:,.2f}")
    print(f"Estimated Annual Savings from Error Reduction: ${annual_savings:,.2f}")
    print(f"Total Costs (Development + Maintenance): ${total_cost:,.2f}")
    print(f"Net Financial Benefit: ${net_benefit:,.2f}")
    
    if net_benefit > 0:
        print("The forecast is financially viable.")
    else:
        print("The forecast may not be financially viable based on current assumptions.")
    
    return annual_savings, total_cost

# Forecasting function
def forecast_sales(model, store, date, holiday_flag, temperature, 
                  fuel_price, cpi, unemployment, previous_sales, features):
    future_date = pd.to_datetime(date)
    holiday_month = holiday_flag * future_date.month
    input_data = pd.DataFrame({
        'Store': [store],
        'Holiday_Flag': [holiday_flag],
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'CPI': [cpi],
        'Unemployment': [unemployment],
        'Year': [future_date.year],
        'Month': [future_date.month],
        'Day': [future_date.day],
        'Lag_1': [previous_sales[0]],
        'Lag_2': [previous_sales[1]],
        'Holiday_Month': [holiday_month]
    })[features]
    
    prediction = model.predict(input_data)[0]
    return prediction

# Main execution
print("Initial Data Processing:")
df_processed = preprocess_data(df)
features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 
           'Unemployment', 'Year', 'Month', 'Day', 'Lag_1', 'Lag_2', 'Holiday_Month']
target = 'Weekly_Sales'

print("\nRemoving Influential Points:")
df_clean = remove_influential_points(df_processed, features, target)

print("\nBuilding Forecast Model with A/B Testing:")
model, y_test, y_pred = build_forecast_model(df_clean, features, target)

# Baseline predictions for cost-benefit analysis
X_train, X_test, y_train, y_test_full = train_test_split(df_clean[features], df_clean[target], test_size=0.2, random_state=42)
baseline_preds = y_test_full.groupby(df_clean.loc[X_test.index, 'Store']).transform('mean')

print("\nPerforming Cost-Benefit Analysis:")
cost_benefit_analysis(df_clean, y_test_full, y_pred, baseline_preds)

# Testing my model
store_id = 1
forecast_date = '2025-03-17'
last_sales = df[df['Store'] == store_id]['Weekly_Sales'].tail(2).values
example_forecast = forecast_sales(
    model, store_id, forecast_date, 
    holiday_flag=0, temperature=60.0, fuel_price=3.5, cpi=220.0, 
    unemployment=7.0, previous_sales=last_sales, features=features
)
print(f"\nForecast for Store {store_id} on {forecast_date}: ${example_forecast:,.2f}")
