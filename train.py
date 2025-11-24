import pandas as pd
import joblib
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

CATEGORICAL_COLS = [
    'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
    'site_category', 'app_id', 'app_domain', 'app_category',
    'device_id', 'device_ip', 'device_model', 'device_type'
]

# Limit data to avoid memory issues
MAX_ROWS = 2000000  # 2M rows - adjust based on your RAM
BATCH_SIZE = 100000  # Process 100K rows at a time

print("Training with batch processing...")
print(f"Max rows: {MAX_ROWS:,} | Batch size: {BATCH_SIZE:,}\n")

hasher = FeatureHasher(n_features=2**18, input_type='dict')
model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, random_state=42)

for epoch in range(2):
    print(f"Epoch {epoch+1}/2")
    rows_processed = 0
    
    for chunk in pd.read_csv('train.csv', chunksize=BATCH_SIZE):
        if rows_processed >= MAX_ROWS:
            break
            
        X_chunk = chunk[CATEGORICAL_COLS].astype(str).to_dict('records')
        y_chunk = chunk['click'].values
        X_hashed = hasher.transform(X_chunk)
        
        model.partial_fit(X_hashed, y_chunk, classes=[0, 1])
        rows_processed += len(chunk)
        print(f"  Processed {rows_processed:,} rows", end='\r')
    
    print(f"  Completed epoch {epoch+1} with {rows_processed:,} rows")

print("\nSaving model...")
joblib.dump(model, 'ctr_model.joblib')
joblib.dump(hasher, 'ctr_hasher.joblib')
print("✓ Model saved successfully!")
print(f"✓ Trained on {rows_processed:,} samples")
