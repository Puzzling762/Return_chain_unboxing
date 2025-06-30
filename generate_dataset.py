import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import timedelta

fake = Faker()
np.random.seed(100)
random.seed(100)

num_rows = 15000

# --- Realistic category distributions ---
product_types = ['Smartphone', 'Laptop', 'Headphones', 'Smartwatch', 'Tablet']
product_probs = [0.35, 0.25, 0.15, 0.15, 0.10]

suppliers = ['Supplier_A', 'Supplier_B', 'Supplier_C', 'Supplier_D', 'Supplier_E']
supplier_quality = {'Supplier_A': 1.5, 'Supplier_B': 3.0, 'Supplier_C': 5.0, 'Supplier_D': 2.0, 'Supplier_E': 4.0}

shipping_carriers = ['Carrier_X', 'Carrier_Y', 'Carrier_Z', 'Carrier_W']
carrier_reliability = {'Carrier_X': 0.9, 'Carrier_Y': 0.7, 'Carrier_Z': 0.8, 'Carrier_W': 0.6}

customer_regions = ['North', 'South', 'East', 'West', 'Central']
region_popularity = [0.25, 0.20, 0.20, 0.20, 0.15]

customer_segments = ['Regular', 'Premium', 'Wholesale']
segment_probs = [0.65, 0.25, 0.10]

transport_modes = ['Air', 'Ground', 'Sea']
transport_probs_map = {  # depends on product type, roughly
    'Smartphone': [0.6, 0.35, 0.05],
    'Laptop': [0.5, 0.45, 0.05],
    'Headphones': [0.3, 0.65, 0.05],
    'Smartwatch': [0.5, 0.45, 0.05],
    'Tablet': [0.4, 0.55, 0.05]
}

return_reasons = ['Defective', 'Late Delivery', 'Wrong Product', 'Changed Mind', 'Other']
return_probs = [0.3, 0.25, 0.1, 0.2, 0.15]

inspection_results = ['Pass', 'Fail', None]
inspection_probs = [0.85, 0.1, 0.05]

# --- Helper to simulate seasonality in orders ---
def seasonality_factor(date):
    month = date.month
    # Higher sales in Nov, Dec (festive), and June-July (mid-year sales)
    if month in [11, 12]:
        return 1.4
    elif month in [6,7]:
        return 1.2
    elif month in [1,2]:
        return 0.8
    else:
        return 1.0

# --- Generate base data ---
product_choices = np.random.choice(product_types, num_rows, p=product_probs)
customer_segments_choices = np.random.choice(customer_segments, num_rows, p=segment_probs)
customer_regions_choices = np.random.choice(customer_regions, num_rows, p=region_popularity)

# SKU and Price with skewed lognormal
price_map = {
    'Smartphone': (400, 1200),
    'Laptop': (600, 2000),
    'Headphones': (50, 300),
    'Smartwatch': (150, 600),
    'Tablet': (200, 900)
}
prices = []
for pt in product_choices:
    low, high = price_map[pt]
    price = np.random.lognormal(mean=np.log((low+high)/2), sigma=0.4)
    price = max(low, min(price, high))
    prices.append(round(price,2))

# Assign supplier weighted by frequency
supplier_weights = [30, 25, 15, 20, 10]
suppliers_list = random.choices(suppliers, weights=supplier_weights, k=num_rows)

# Assign transport modes based on product type
transport_modes_list = []
for pt in product_choices:
    transport_modes_list.append(np.random.choice(transport_modes, p=transport_probs_map[pt]))

# Assign shipping carrier based on price and reliability bias
def choose_carrier(price):
    if price > 1000:
        weights = [0.4, 0.3, 0.2, 0.1]
    elif price > 500:
        weights = [0.3, 0.4, 0.2, 0.1]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
    return random.choices(shipping_carriers, weights=weights, k=1)[0]

shipping_carriers_list = [choose_carrier(p) for p in prices]

# Generate Order Dates with seasonality over last 12 months
base_date = pd.to_datetime('today') - pd.DateOffset(years=1)
order_dates = []
for _ in range(num_rows):
    # Random day offset in past year
    random_days = np.random.randint(0, 365)
    date = base_date + pd.Timedelta(days=random_days)
    order_dates.append(date)

# --- UPDATED Number Products Sold with product multipliers (smartwatch less) and seasonality ---
base_sales_mean = {'Regular': 15, 'Premium': 25, 'Wholesale': 40}

product_sales_multiplier = {
    'Smartphone': 1.5,
    'Laptop': 1.2,
    'Headphones': 0.9,
    'Smartwatch': 0.6,   # lower sales
    'Tablet': 0.8
}

number_products_sold = []
for seg, pt, date in zip(customer_segments_choices, product_choices, order_dates):
    mean_sales = base_sales_mean[seg] * product_sales_multiplier[pt]
    mean_sales *= seasonality_factor(date)
    sold = np.random.poisson(mean_sales)
    number_products_sold.append(max(0, sold))

# Revenue
revenue_generated = [p*s for p,s in zip(prices, number_products_sold)]

# Availability by product type (more out of stock during high demand)
avail_probs_map = {
    'Smartphone': 0.85,
    'Laptop': 0.8,
    'Headphones': 0.9,
    'Smartwatch': 0.85,
    'Tablet': 0.8
}
availability = []
for pt in product_choices:
    availability.append(np.random.choice(['In Stock', 'Out of Stock'], p=[avail_probs_map[pt], 1-avail_probs_map[pt]]))

# Lead times: supplier quality, carrier reliability, transport mode influence
lead_times = []
manufacturing_lead_times = []
shipping_times = []
for sup, carrier, mode in zip(suppliers_list, shipping_carriers_list, transport_modes_list):
    base_lead = np.random.randint(3, 10)
    # Supplier quality: better supplier -> shorter lead times
    supplier_factor = max(1, 5 - supplier_quality[sup]/2)
    # Carrier reliability affects shipping time
    carrier_factor = 1 if carrier_reliability[carrier] > 0.8 else 1.5
    # Transport mode typical days
    if mode == 'Air':
        mode_time = np.random.randint(1, 3)
    elif mode == 'Ground':
        mode_time = np.random.randint(3, 7)
    else:
        mode_time = np.random.randint(10, 20)
    manufacturing_lead = int(np.random.normal(7, 2))
    manufacturing_lead = max(3, manufacturing_lead)
    shipping_time = int(mode_time * carrier_factor)
    lead_time = int(base_lead * supplier_factor + shipping_time)
    lead_times.append(lead_time)
    manufacturing_lead_times.append(manufacturing_lead)
    shipping_times.append(shipping_time)

# Supplier defect rates + noise
supplier_defect_rate = []
for sup in suppliers_list:
    base_defect = supplier_quality[sup]
    noise = np.random.normal(0, 0.5)
    val = max(0.5, min(10, base_defect + noise))
    supplier_defect_rate.append(round(val, 2))

# Inspection results with some missingness related to product & supplier
inspection_results_list = []
for pt, sup in zip(product_choices, suppliers_list):
    if pt == 'Headphones' and random.random() < 0.2:
        inspection_results_list.append(None)
    elif sup == 'Supplier_C' and random.random() < 0.3:
        inspection_results_list.append(None)
    else:
        inspection_results_list.append(np.random.choice(inspection_results, p=inspection_probs))

# Defect rate linked to supplier defect and inspection results
defect_rate = []
for sup_def, insp in zip(supplier_defect_rate, inspection_results_list):
    base = sup_def
    if insp == 'Fail':
        base += np.random.uniform(1,3)
    elif insp == 'Pass':
        base -= np.random.uniform(0,0.5)
    defect_rate.append(round(max(0.1, base), 2))

# Return reason only if defect rate high or late delivery, else small chance
return_reason = []
return_flags = []
for dr in defect_rate:
    if dr > 5:
        # High chance of return due to defect
        return_flags.append(True if random.random() < 0.6 else False)
    elif random.random() < 0.12:
        # Random returns for other reasons
        return_flags.append(True)
    else:
        return_flags.append(False)

for flag in return_flags:
    if flag:
        reason = np.random.choice(return_reasons, p=return_probs)
    else:
        reason = None
    return_reason.append(reason)

# Return date after order date (within 30 days)
return_date = []
for r, o_date in zip(return_reason, order_dates):
    if r is not None:
        ret_date = o_date + timedelta(days=np.random.randint(1, 30))
        return_date.append(ret_date)
    else:
        return_date.append(None)

# Refund status depends on return reason
refund_status = []
for rr in return_reason:
    if rr is None:
        refund_status.append(None)
    else:
        refund_status.append(np.random.choice(['Completed', 'Pending', 'Rejected'], p=[0.6, 0.3, 0.1]))

# Shipping cost correlated with price and mode (air costlier)
shipping_cost = []
for p, mode in zip(prices, transport_modes_list):
    base = p * np.random.uniform(0.03, 0.06)
    if mode == 'Air':
        base *= 1.5
    elif mode == 'Sea':
        base *= 0.6
    shipping_cost.append(round(base, 2))

# Notes with missingness, more missing for wholesale and headphones
notes = []
for seg, pt in zip(customer_segments_choices, product_choices):
    prob_note = 0.1
    if seg == 'Wholesale':
        prob_note = 0.05
    if pt == 'Headphones':
        prob_note -= 0.03
    notes.append(fake.sentence() if random.random() < prob_note else None)

# Create DataFrame
df = pd.DataFrame({
    'SKU': [f'SKU{10000 + i}' for i in range(num_rows)],
    'Order_Date': order_dates,
    'Product_Type': product_choices,
    'Price': prices,
    'Availability': availability,
    'Number_Products_Sold': number_products_sold,
    'Revenue_Generated': revenue_generated,
    'Customer_Region': customer_regions_choices,
    'Customer_Segment': customer_segments_choices,
    'Supplier_Name': suppliers_list,
    'Supplier_Defect_Rate': supplier_defect_rate,
    'Inspection_Result': inspection_results_list,
    'Defect_Rate': defect_rate,
    'Return_Reason': return_reason,
    'Return_Date': return_date,
    'Refund_Status': refund_status,
    'Shipping_Carrier': shipping_carriers_list,
    'Shipping_Cost': shipping_cost,
    'Transport_Mode': transport_modes_list,
    'Lead_Time': lead_times,
    'Manufacturing_Lead_Time': manufacturing_lead_times,
    'Shipping_Time': shipping_times,
    'Notes': notes,
})

print(df.sample(5))
df.to_csv('data.csv', index=False)
print("Enhanced realistic dummy dataset saved as data.csv")
