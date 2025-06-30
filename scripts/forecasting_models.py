import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
from datetime import datetime

def run_forecasting_engine_powerbi(df_clean, forecast_periods=6, save_plots=True, output_dir='forecast_output'):
    """
    Compact Forecasting Engine with Power BI-Ready CSV Export
    
    Returns: dict with 'powerbi_data' (DataFrame) and summary stats
    """
    
    def get_revenue_column(df):
        revenue_cols = ['Revenue_Generated', 'Price', 'Product_Price', 'Number_Products_Sold', 'Quantity', 'Sales']
        for col in revenue_cols:
            if col in df.columns:
                return col
        raise ValueError("No revenue column found")
    
    def prepare_data(df):
        df = df.copy()
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        df['Order_Month'] = df['Order_Date'].dt.to_period('M')
        revenue_col = get_revenue_column(df)
        
        monthly = df.groupby(['Product_Type', 'Order_Month']).agg({revenue_col: 'sum'}).reset_index()
        monthly.columns = ['Product_Type', 'Order_Month', 'Monthly_Sales']
        return monthly, revenue_col
    
    def arima_forecast(ts, steps=6):
        if len(ts) < 6:
            return None
        
        d = 0
        ts_test = ts.copy()
        while d <= 2:
            try:
                if adfuller(ts_test.dropna())[1] <= 0.05:
                    break
            except:
                pass
            d += 1
            ts_test = ts_test.diff().dropna()
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = ARIMA(ts, order=(1, d, 1))
                fitted = model.fit()
                forecast = fitted.forecast(steps=steps)
                return np.maximum(forecast, 0)
        except:
            return None
    
    def prophet_forecast(df, product, periods=6):
        data = df[df['Product_Type'] == product].copy()
        if len(data) < 6:
            return None
        
        prophet_df = pd.DataFrame({
            'ds': data['Order_Month'].dt.to_timestamp(),
            'y': data['Monthly_Sales']
        })
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
                model.fit(prophet_df)
                future = model.make_future_dataframe(periods=periods, freq='MS')
                forecast = model.predict(future)
                return np.maximum(forecast['yhat'].tail(periods).values, 0)
        except:
            return None
    
    def simple_baselines(ts, steps=6):
        naive = np.full(steps, ts.iloc[-1])
        moving_avg = np.full(steps, ts.tail(6).mean())
        trend = np.linspace(ts.iloc[-1], ts.iloc[-1] + (ts.diff().tail(3).mean() * steps), steps)
        return {'naive': naive, 'moving_avg': moving_avg, 'trend': np.maximum(trend, 0)}
    
    def evaluate_forecast(ts, forecast_func, **kwargs):
        if len(ts) < 12:
            return {'MAE': 0, 'RMSE': 0, 'MAPE': 0}
        
        test_size = min(3, len(ts) // 4)
        train_data, test_data = ts.iloc[:-test_size], ts.iloc[-test_size:]
        
        try:
            if forecast_func == arima_forecast:
                pred = arima_forecast(train_data, steps=test_size)
            elif forecast_func == prophet_forecast:
                pred = prophet_forecast(kwargs['df'], kwargs['product'], periods=test_size)
            else:
                pred = forecast_func(train_data, steps=test_size)[kwargs['method']]
            
            if pred is not None:
                mae = mean_absolute_error(test_data, pred)
                rmse = np.sqrt(mean_squared_error(test_data, pred))
                mape_values = [abs((a - p) / a) for a, p in zip(test_data, pred) if a != 0]
                mape = np.mean(mape_values) * 100 if mape_values else 0
                return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        except:
            pass
        return {'MAE': 0, 'RMSE': 0, 'MAPE': 0}
    
    def create_plot(ts, forecasts, product):
        plt.figure(figsize=(12, 6))
        plt.plot(ts.index, ts.values, 'o-', label='Historical', color='blue', linewidth=2)
        
        future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i, (name, forecast) in enumerate(forecasts.items()):
            if forecast is not None:
                plt.plot(future_dates, forecast, 's-', label=name.title(), color=colors[i % len(colors)], alpha=0.8)
        
        plt.axvline(x=ts.index[-1], color='gray', linestyle='--', alpha=0.5, label='Forecast Start')
        plt.title(f'Sales Forecast - {product}', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Sales Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    # Main execution
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    monthly_data, revenue_column = prepare_data(df_clean)
    products = monthly_data['Product_Type'].unique()
    powerbi_data = []
    
    print(f"🚀 Forecasting {len(products)} products for {forecast_periods} months")
    print(f"📊 Using revenue column: {revenue_column}")
    
    for product in products:
        print(f"\n📊 Processing: {product}")
        
        product_data = monthly_data[monthly_data['Product_Type'] == product].copy()
        if len(product_data) < 6:
            print(f"   ⚠️ Insufficient data - skipping")
            continue
        
        product_data['Order_Month'] = product_data['Order_Month'].dt.to_timestamp()
        product_data.set_index('Order_Month', inplace=True)
        ts = product_data['Monthly_Sales'].asfreq('MS').fillna(method='ffill')
        
        # Calculate metrics
        growth_rate = ts.pct_change().dropna().mean() * 100 if len(ts) >= 2 else 0
        seasonality_score = (ts.std() / ts.mean()) * 100 if ts.mean() != 0 else 0
        volatility = seasonality_score
        
        # Generate forecasts
        arima_pred = arima_forecast(ts, forecast_periods)
        prophet_pred = prophet_forecast(monthly_data, product, forecast_periods)
        baselines = simple_baselines(ts, forecast_periods)
        
        forecasts = {
            'ARIMA': arima_pred,
            'Prophet': prophet_pred,
            'Naive': baselines['naive'],
            'Moving_Avg': baselines['moving_avg'],
            'Trend': baselines['trend']
        }
        
        # Evaluate and find best model
        models_eval = {
            'ARIMA': evaluate_forecast(ts, arima_forecast),
            'Prophet': evaluate_forecast(ts, prophet_forecast, df=monthly_data, product=product),
            'Naive': evaluate_forecast(ts, simple_baselines, method='naive'),
            'Moving_Avg': evaluate_forecast(ts, simple_baselines, method='moving_avg'),
            'Trend': evaluate_forecast(ts, simple_baselines, method='trend')
        }
        
        valid_models = {k: v for k, v in models_eval.items() if v['MAE'] > 0}
        if valid_models:
            best_name = min(valid_models.items(), key=lambda x: x[1]['MAE'])[0]
            best_metrics = models_eval[best_name]
        else:
            best_name, best_metrics = 'Naive', {'MAE': 0, 'RMSE': 0, 'MAPE': 0}
        
        print(f"   🏆 Best: {best_name} (MAE: {best_metrics['MAE']:.1f})")
        
        # Create and save plot
        create_plot(ts, forecasts, product)
        if save_plots:
            plt.savefig(f'{output_dir}/forecast_{product.replace(" ", "_").replace("/", "_")}.png', dpi=300, bbox_inches='tight')
            print(f"   💾 Plot saved")
        
        # Add historical data to Power BI dataset
        for idx, (date, value) in enumerate(ts.items()):
            powerbi_data.append({
                'Product_Type': product,
                'Date': date.strftime('%Y-%m-%d'),
                'Year': date.year,
                'Month': date.month,
                'Month_Name': date.strftime('%B'),
                'Quarter': f"Q{date.quarter}",
                'Year_Month': date.strftime('%Y-%m'),
                'Data_Type': 'Historical',
                'Is_Forecast': False,
                'Is_Historical': True,
                'Actual_Value': round(value, 2),
                'Forecast_Value': None,
                'Best_Model_Forecast': None,
                'ARIMA_Forecast': None,
                'Prophet_Forecast': None,
                'Naive_Forecast': None,
                'Moving_Avg_Forecast': None,
                'Trend_Forecast': None,
                'Best_Model': best_name,
                'Model_MAE': round(best_metrics['MAE'], 2),
                'Model_RMSE': round(best_metrics['RMSE'], 2),
                'Model_MAPE': round(best_metrics['MAPE'], 2),
                'Data_Points_Available': len(ts),
                'Growth_Rate_Percent': round(growth_rate, 2),
                'Seasonality_Score': round(seasonality_score, 2),
                'Volatility_Percent': round(volatility, 2),
                'Min_Historical_Value': round(ts.min(), 2),
                'Max_Historical_Value': round(ts.max(), 2),
                'Avg_Historical_Value': round(ts.mean(), 2),
                'Last_Historical_Value': round(ts.iloc[-1], 2),
                'Forecast_Periods': forecast_periods,
                'Revenue_Column_Used': revenue_column,
                'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Period_Number': idx + 1,
                'Is_Recent_3_Months': idx >= len(ts) - 3,
                'Is_Recent_6_Months': idx >= len(ts) - 6,
                'Is_Recent_12_Months': idx >= len(ts) - 12,
                'Forecast_Confidence': 'High' if best_metrics['MAPE'] < 10 else 'Medium' if best_metrics['MAPE'] < 25 else 'Low'
            })
        
        # Add forecast data to Power BI dataset
        if forecasts[best_name] is not None:
            future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
            
            for i, date in enumerate(future_dates):
                powerbi_data.append({
                    'Product_Type': product,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Year': date.year,
                    'Month': date.month,
                    'Month_Name': date.strftime('%B'),
                    'Quarter': f"Q{date.quarter}",
                    'Year_Month': date.strftime('%Y-%m'),
                    'Data_Type': 'Forecast',
                    'Is_Forecast': True,
                    'Is_Historical': False,
                    'Actual_Value': None,
                    'Forecast_Value': round(forecasts[best_name][i], 2),
                    'Best_Model_Forecast': round(forecasts[best_name][i], 2),
                    'ARIMA_Forecast': round(forecasts['ARIMA'][i], 2) if forecasts['ARIMA'] is not None else None,
                    'Prophet_Forecast': round(forecasts['Prophet'][i], 2) if forecasts['Prophet'] is not None else None,
                    'Naive_Forecast': round(forecasts['Naive'][i], 2),
                    'Moving_Avg_Forecast': round(forecasts['Moving_Avg'][i], 2),
                    'Trend_Forecast': round(forecasts['Trend'][i], 2),
                    'Best_Model': best_name,
                    'Model_MAE': round(best_metrics['MAE'], 2),
                    'Model_RMSE': round(best_metrics['RMSE'], 2),
                    'Model_MAPE': round(best_metrics['MAPE'], 2),
                    'Data_Points_Available': len(ts),
                    'Growth_Rate_Percent': round(growth_rate, 2),
                    'Seasonality_Score': round(seasonality_score, 2),
                    'Volatility_Percent': round(volatility, 2),
                    'Min_Historical_Value': round(ts.min(), 2),
                    'Max_Historical_Value': round(ts.max(), 2),
                    'Avg_Historical_Value': round(ts.mean(), 2),
                    'Last_Historical_Value': round(ts.iloc[-1], 2),
                    'Forecast_Periods': forecast_periods,
                    'Revenue_Column_Used': revenue_column,
                    'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Period_Number': len(ts) + i + 1,
                    'Is_Recent_3_Months': False,
                    'Is_Recent_6_Months': False,
                    'Is_Recent_12_Months': False,
                    'Forecast_Period_Number': i + 1,
                    'Forecast_Confidence': 'High' if best_metrics['MAPE'] < 10 else 'Medium' if best_metrics['MAPE'] < 25 else 'Low'
                })
    
    # Create Power BI DataFrame
    powerbi_df = pd.DataFrame(powerbi_data).sort_values(['Product_Type', 'Date']).reset_index(drop=True)
    
    # Add YoY Growth calculation
    if not powerbi_df.empty:
        powerbi_df['YoY_Growth'] = None
        for product in powerbi_df['Product_Type'].unique():
            product_data = powerbi_df[powerbi_df['Product_Type'] == product]
            for idx, row in product_data.iterrows():
                if row['Is_Historical']:
                    current_date = pd.to_datetime(row['Date'])
                    prev_year_date = current_date - pd.DateOffset(years=1)
                    prev_mask = ((powerbi_df['Product_Type'] == product) & 
                               (pd.to_datetime(powerbi_df['Date']) == prev_year_date) &
                               (powerbi_df['Is_Historical'] == True))
                    
                    if prev_mask.any():
                        prev_value = powerbi_df[prev_mask]['Actual_Value'].iloc[0]
                        if prev_value and prev_value != 0:
                            yoy_growth = ((row['Actual_Value'] - prev_value) / prev_value) * 100
                            powerbi_df.loc[idx, 'YoY_Growth'] = round(yoy_growth, 2)
    
    # Save the Power BI file
    if save_plots and not powerbi_df.empty:
        powerbi_filename = f'{output_dir}/PowerBI_Forecast_Complete.csv'
        powerbi_df.to_csv(powerbi_filename, index=False)
        print(f"\n🎯 POWER BI FILE SAVED: {powerbi_filename}")
        print(f"📊 Total records: {len(powerbi_df)} | Products: {len(powerbi_df['Product_Type'].unique())}")
        print(f"📅 Date range: {powerbi_df['Date'].min()} to {powerbi_df['Date'].max()}")
        print(f"🖼️ Charts saved: {len(products)} forecast plots")
    
    plt.show()
    
    return {
        'powerbi_data': powerbi_df,
        'summary_stats': {
            'total_products': len(products),
            'total_records': len(powerbi_df),
            'forecast_periods': forecast_periods,
            'revenue_column': revenue_column,
            'output_file': f'{output_dir}/PowerBI_Forecast_Complete.csv' if save_plots else None
        }
    }

# Example usage:
# results = run_forecasting_engine_powerbi(df_clean, forecast_periods=6)
# powerbi_df = results['powerbi_data']