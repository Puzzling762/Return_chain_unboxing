
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


FESTIVAL_PATTERNS = {
    '2024-09': 1.25, '2024-10': 1.60, '2024-11': 1.70, '2024-12': 1.35,
    '2025-01': 0.83, '2025-02': 0.95, '2025-03': 1.00, '2025-04': 1.18,
    '2025-05': 1.05, '2025-06': 0.95, '2025-07': 0.88, '2025-08': 1.13
}

DEFECT_MULTIPLIERS = {
    '2024-09': 1.1, '2024-10': 1.4, '2024-11': 1.6, '2024-12': 1.3,
    '2025-01': 0.8, '2025-02': 0.9, '2025-03': 1.0, '2025-04': 1.1,
    '2025-05': 1.2, '2025-06': 1.0, '2025-07': 1.3, '2025-08': 1.1
}

LEAD_TIME_ADJUSTMENTS = {
    '2024-09': 3, '2024-10': 5, '2024-11': 8, '2024-12': 6,
    '2025-01': 2, '2025-02': 1, '2025-03': 2, '2025-04': 3,
    '2025-05': 4, '2025-06': 3, '2025-07': 9, '2025-08': 5
}

def auto_detect_columns(df):
    """Automatically detect required columns"""
    cols = {'date': None, 'product': None, 'sales': None, 'lead_time': None}
    
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['date', 'order']) and cols['date'] is None:
            try:
                pd.to_datetime(df[col])
                cols['date'] = col
            except: pass
        elif any(word in col_lower for word in ['product', 'type', 'category']) and cols['product'] is None:
            cols['product'] = col
        elif any(word in col_lower for word in ['sales', 'sold', 'quantity', 'qty', 'units']) and cols['sales'] is None:
            if df[col].dtype in ['int64', 'float64']:
                cols['sales'] = col
        elif any(word in col_lower for word in ['lead', 'time', 'delivery']) and cols['lead_time'] is None:
            if df[col].dtype in ['int64', 'float64']:
                cols['lead_time'] = col
    
    return cols

def prepare_monthly_data(df, date_col=None, product_col=None, sales_col=None, lead_time_col=None):
    """Prepare data for simulation"""
    print("🔍 Preparing data...")
    
 
    if not all([date_col, product_col, sales_col]):
        detected = auto_detect_columns(df)
        date_col = date_col or detected['date']
        product_col = product_col or detected['product']
        sales_col = sales_col or detected['sales']
        lead_time_col = lead_time_col or detected['lead_time']
    
    if not all([date_col, product_col, sales_col]):
        print("❌ Could not detect required columns. Please specify manually.")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    df_work = df.copy()
    df_work[date_col] = pd.to_datetime(df_work[date_col])
    df_work['Order_Month'] = df_work[date_col].dt.to_period('M').astype(str)
    
 
    agg_dict = {sales_col: 'sum'}
    if lead_time_col:
        agg_dict[lead_time_col] = 'mean'
    
    monthly_df = df_work.groupby(['Order_Month', product_col], as_index=False).agg(agg_dict)
    

    monthly_df = monthly_df.rename(columns={
        product_col: 'Product_Type',
        sales_col: 'Monthly_Sales'
    })
    
    if lead_time_col:
        monthly_df = monthly_df.rename(columns={lead_time_col: 'Lead_Time'})
        monthly_df['Lead_Time'] = monthly_df['Lead_Time'].round().astype(int)
    else:
        monthly_df['Lead_Time'] = np.random.choice([3, 5, 7, 10], len(monthly_df))
    
    print(f"✅ Data prepared: {monthly_df.shape} records, {len(monthly_df['Product_Type'].unique())} products")
    return monthly_df

def run_simulation(df, base_defect_rate=0.05, return_cost=3500, lead_time_sla=7):
    """Run complete simulation for all products"""
    np.random.seed(42)
    results = []
    
    for product in df['Product_Type'].unique():
        print(f"🔍 Simulating: {product}")
        
        mask = df['Product_Type'] == product
        product_data = df[mask].copy()
        

        product_data['Simulated_Demand'] = product_data['Monthly_Sales'].astype(float)
        for month in product_data['Order_Month']:
            if month in FESTIVAL_PATTERNS:
                month_mask = product_data['Order_Month'] == month
                product_data.loc[month_mask, 'Simulated_Demand'] *= FESTIVAL_PATTERNS[month]
        
        product_data['Simulated_Demand'] *= np.random.uniform(0.9, 1.2, len(product_data))
        product_data['Simulated_Demand'] = product_data['Simulated_Demand'].round().astype(int)
        

        product_data['Defect_Rate'] = base_defect_rate
        for month in product_data['Order_Month']:
            if month in DEFECT_MULTIPLIERS:
                month_mask = product_data['Order_Month'] == month
                product_data.loc[month_mask, 'Defect_Rate'] *= DEFECT_MULTIPLIERS[month]
        
        product_data['Simulated_Defects'] = np.maximum(
            (product_data['Simulated_Demand'] * product_data['Defect_Rate']).round().astype(int), 1
        )
        

        product_data['Simulated_Lead_Time'] = product_data['Lead_Time'].copy()
        for month in product_data['Order_Month']:
            if month in LEAD_TIME_ADJUSTMENTS:
                month_mask = product_data['Order_Month'] == month
                product_data.loc[month_mask, 'Simulated_Lead_Time'] += LEAD_TIME_ADJUSTMENTS[month]
        
        product_data['Simulated_Lead_Time'] += np.random.randint(-1, 4, len(product_data))
        product_data['Simulated_Lead_Time'] = np.maximum(product_data['Simulated_Lead_Time'], 1)
        

        product_data['Return_Cost'] = product_data['Simulated_Defects'] * return_cost
        product_data['SLA_Breach'] = (product_data['Simulated_Lead_Time'] > lead_time_sla).astype(int)
        product_data['High_Risk'] = (product_data['Return_Cost'] > 100000).astype(int)
        

        defect_impact = product_data['Simulated_Defects'] / product_data['Simulated_Demand'] * 4
        lead_impact = np.maximum(product_data['Simulated_Lead_Time'] - 10, 0) * 0.25
        product_data['Customer_Satisfaction'] = np.maximum(7.5 - defect_impact - lead_impact, 1.0).round(2)
        
        results.append(product_data)
    
    return pd.concat(results, ignore_index=True)

def validate_model(df):
    """Validate simulation accuracy"""
    validation_results = {}
    
    for product in df['Product_Type'].unique():
        mask = df['Product_Type'] == product
        data = df[mask].copy()
        
        if len(data) < 3:
            continue
            

        data['Month_Num'] = pd.to_datetime(data['Order_Month']).dt.month
        data['Festival_Score'] = data['Order_Month'].map(FESTIVAL_PATTERNS).fillna(1.0)
        
        X = data[['Monthly_Sales', 'Lead_Time', 'Month_Num', 'Festival_Score']]
        y = data['Simulated_Demand']
        
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        
        validation_results[product] = {
            'R2_Score': r2_score(y, y_pred),
            'MAPE': mean_absolute_percentage_error(y, y_pred)
        }
    
    return validation_results

def monte_carlo_analysis(df, product, n_simulations=1000):
    """Run Monte Carlo risk analysis"""
    mask = df['Product_Type'] == product
    base_sales = df[mask]['Monthly_Sales'].values
    
    costs = []
    for _ in range(n_simulations):
        demand_factor = np.random.uniform(0.7, 1.5)
        defect_rate = np.random.uniform(0.03, 0.08)
        
        sim_demand = base_sales * demand_factor
        sim_defects = sim_demand * defect_rate
        total_cost = sim_defects.sum() * 3500
        costs.append(total_cost)
    
    return np.array(costs)

def create_visualizations(df, save_plots=True):
    """Generate all visualizations"""
    if not save_plots:
        return
    
    products = df['Product_Type'].unique()
    

    plt.figure(figsize=(15, 8))
    for i, product in enumerate(products[:4]):  
        plt.subplot(2, 2, i+1)
        mask = df['Product_Type'] == product
        data = df[mask]
        
        plt.plot(data['Order_Month'], data['Monthly_Sales'], 'o-', label='Actual Sales', linewidth=2)
        plt.plot(data['Order_Month'], data['Simulated_Demand'], 's-', label='Simulated Demand', linewidth=2)
        plt.title(f'{product} - Demand Analysis')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demand_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    

    plt.figure(figsize=(12, 8))
    risk_matrix = df.pivot_table(values='Return_Cost', index='Product_Type', 
                                columns='Order_Month', aggfunc='sum', fill_value=0)
    plt.imshow(risk_matrix.values, cmap='Reds', aspect='auto')
    plt.colorbar(label='Return Cost (₹)')
    plt.title('Risk Heatmap - Return Costs by Product & Month')
    plt.yticks(range(len(risk_matrix.index)), risk_matrix.index)
    plt.xticks(range(len(risk_matrix.columns)), risk_matrix.columns, rotation=45)
    plt.tight_layout()
    plt.savefig('risk_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
  
    for product in products:
        costs = monte_carlo_analysis(df, product)
        
        plt.figure(figsize=(10, 6))
        plt.hist(costs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(costs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ₹{costs.mean():,.0f}')
        plt.axvline(np.percentile(costs, 95), color='orange', linestyle='--', linewidth=2, 
                   label=f'95th Percentile: ₹{np.percentile(costs, 95):,.0f}')
        plt.title(f'Monte Carlo Risk Analysis - {product}')
        plt.xlabel('Total Return Cost (₹)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'monte_carlo_{product.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_reports(df, validation_results):
    """Generate comprehensive reports"""
    print("\n" + "="*80)
    print("📊 SUPPLY CHAIN SIMULATION REPORT")
    print("="*80)
    
    for product in df['Product_Type'].unique():
        mask = df['Product_Type'] == product
        data = df[mask]
        
        print(f"\n🏷️  PRODUCT: {product}")
        print("-" * 50)
        print(f"📈 Total Simulated Demand: {data['Simulated_Demand'].sum():,}")
        print(f"🔧 Total Defects: {data['Simulated_Defects'].sum():,}")
        print(f"💰 Total Return Cost: ₹{data['Return_Cost'].sum():,.0f}")
        print(f"⏰ SLA Breaches: {data['SLA_Breach'].sum()}")
        print(f"⚠️  High Risk Periods: {data['High_Risk'].sum()}")
        print(f"😊 Avg Customer Satisfaction: {data['Customer_Satisfaction'].mean():.2f}/10")
        
        if product in validation_results:
            val = validation_results[product]
            print(f"🎯 Model R² Score: {val['R2_Score']:.3f}")
            print(f"📊 Model MAPE: {val['MAPE']:.1%}")
            
    
            if val['R2_Score'] > 0.8:
                print("✅ Model Quality: Excellent")
            elif val['R2_Score'] > 0.6:
                print("⚠️  Model Quality: Good")
            else:
                print("❌ Model Quality: Needs Improvement")

def run_simulation_engine(df_input, date_col=None, product_col=None, sales_col=None, 
                         lead_time_col=None, save_plots=True, base_defect_rate=0.05):
    """
    Main function to run complete simulation
    
    Usage:
    results = run_simulation_engine(df_clean, save_plots=True)
    
    Or with manual column specification:
    results = run_simulation_engine(df_clean, 
                                   date_col='Order_Date',
                                   product_col='Product_Type',
                                   sales_col='Number_Products_Sold',
                                   save_plots=True)
    """
    print("🚀 Starting Enhanced Supply Chain Simulation...")
    
 
    monthly_df = prepare_monthly_data(df_input, date_col, product_col, sales_col, lead_time_col)
    if monthly_df is None:
        return None
    
   
    print("⚙️  Running simulation...")
    results_df = run_simulation(monthly_df, base_defect_rate=base_defect_rate)
    
   
    print("🎯 Validating model...")
    validation_results = validate_model(results_df)
    
   
    if save_plots:
        print("📊 Creating visualizations...")
        create_visualizations(results_df, save_plots=True)
    
    
    print("💾 Exporting results...")
    
   
    results_df.to_csv('supply_chain_simulation_results.csv', index=False)
    
   
    summary = results_df.groupby('Product_Type').agg({
        'Monthly_Sales': 'sum',
        'Simulated_Demand': 'sum',
        'Simulated_Defects': 'sum',
        'Return_Cost': 'sum',
        'SLA_Breach': 'sum',
        'High_Risk': 'sum',
        'Customer_Satisfaction': 'mean'
    }).round(2)
    summary.to_csv('product_summary.csv')
    
    
    monthly_summary = results_df.groupby('Order_Month').agg({
        'Simulated_Demand': 'sum',
        'Simulated_Defects': 'sum',
        'Return_Cost': 'sum',
        'SLA_Breach': 'sum',
        'Customer_Satisfaction': 'mean'
    }).round(2)
    monthly_summary.to_csv('monthly_summary.csv')
    
   
    generate_reports(results_df, validation_results)
    
    print("\n🎉 Simulation Complete!")
    print("📁 Files Generated:")
    print("   ✅ supply_chain_simulation_results.csv - Complete results")
    print("   ✅ product_summary.csv - Summary by product")
    print("   ✅ monthly_summary.csv - Summary by month")
    if save_plots:
        print("   ✅ demand_analysis.png - Demand vs actual comparison")
        print("   ✅ risk_heatmap.png - Risk visualization")
        print("   ✅ monte_carlo_[product].png - Risk distribution for each product")
    
    return results_df
