import pandas as pd
import matplotlib.pyplot as plt

print("Loading data...")
df = pd.read_csv('train.csv', nrows=100000)

print("\n=== Dataset Info ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

print("\n=== Click Distribution ===")
click_dist = df['click'].value_counts()
print(click_dist)
print(f"\nCTR: {df['click'].mean():.4f}")

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Categorical Features Cardinality ===")
cat_cols = ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 
            'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 
            'device_model', 'device_type']
for col in cat_cols:
    if col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

print("\n=== CTR by Device Type ===")
if 'device_type' in df.columns:
    print(df.groupby('device_type')['click'].agg(['count', 'mean']))

print("\n=== CTR by Banner Position ===")
if 'banner_pos' in df.columns:
    print(df.groupby('banner_pos')['click'].agg(['count', 'mean']))

print("\nEDA Complete!")
