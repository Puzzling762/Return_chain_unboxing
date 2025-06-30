
import pandas as pd
import numpy as np

def load_raw_data(filepath):
    """
    Loads raw data from CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning and preprocessing.
    """
    initial_count = len(df)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df['Return_Date'] = pd.to_datetime(df['Return_Date'], errors='coerce')

    df = df.dropna(subset=['Order_Date'])
    dropped = initial_count - len(df)
    print(f"Dropped {dropped} rows due to missing or invalid Order_Date")

    df['Defect_Rate'] = df['Defect_Rate'].fillna(0)
    df['Return_Reason'] = df['Return_Reason'].fillna('Unknown')

    numeric_cols = ['Price', 'Number_Products_Sold', 'Revenue_Generated', 'Lead_Time',
                    'Manufacturing_Lead_Time', 'Shipping_Time', 'Shipping_Cost',
                    'Supplier_Defect_Rate', 'Defect_Rate']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


    df['Order_Month'] = df['Order_Date'].dt.to_period('M')

    return df

def aggregate_monthly_sales(df):
    """
    Aggregates monthly sales, returns and defect rate by product type.
    """

    sales = df.groupby(['Order_Month', 'Product_Type'])['Number_Products_Sold'].sum().reset_index()


    df['Is_Return'] = df['Return_Date'].notnull().astype(int)
    returns = df.groupby(['Order_Month', 'Product_Type'])['Is_Return'].sum().reset_index()

    defect_rate = df.groupby(['Order_Month', 'Product_Type'])['Defect_Rate'].mean().reset_index()

    monthly_summary = sales.merge(returns, on=['Order_Month', 'Product_Type'])
    monthly_summary = monthly_summary.merge(defect_rate, on=['Order_Month', 'Product_Type'])
    monthly_summary.rename(columns={'Number_Products_Sold': 'Monthly_Sales', 'Is_Return': 'Monthly_Returns', 'Defect_Rate': 'Avg_Defect_Rate'}, inplace=True)

    return monthly_summary

if __name__ == "__main__":
    df = load_raw_data('returns data analysis\data\data.csv')
    df_clean = preprocess_data(df)
    monthly_summary = aggregate_monthly_sales(df_clean)
    print(monthly_summary.head())