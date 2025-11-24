import joblib

# this is predict.py is only for demonstration purposes. for initial deployment 



# --- 1. LOAD THE SAVED MODEL AND HASHER ---
print("Loading model and hasher...")
try:
    model = joblib.load('ctr_model.joblib')
    hasher = joblib.load('ctr_hasher.joblib')
except FileNotFoundError:
    print("Error: Model files not found.")
    print("Please run train.py first to create the model files.")
    exit()

print("Model and hasher loaded successfully.")

# --- 2. MAKE A PREDICTION ON NEW DATA ---
print("\n--- Example Prediction ---")

# Let's create two new ad impressions to predict
new_ad_data = [
    { # Example 1: The one from our test (should be low prob)
        'hour': '14102102', 'C1': '1005', 'banner_pos': '1', 'site_id': '510bdc79',
        'site_domain': '30d31379', 'site_category': 'f028772b', 'app_id': 'ecad2386',
        'app_domain': '7801e8d9', 'app_category': '07d7df22', 'device_id': 'a99f214a',
        'device_ip': '1575c2df', 'device_model': 'e5801f65', 'device_type': '1'
    },
    { # Example 2: A different ad (e.g., on a mobile app)
        'hour': '14102103', 'C1': '1002', 'banner_pos': '0', 'site_id': '85f751fd',
        'site_domain': 'c4e18dd6', 'site_category': '50e219e0', 'app_id': 'e2fcccd2',
        'app_domain': '5c5a694b', 'app_category': '0f2161f8', 'device_id': 'a99f214a',
        'device_ip': '17113110', 'device_model': '0707b4bc', 'device_type': '1'
    }
]

# 1. Transform the new data using the *loaded* hasher
# Note: We must convert all values to string first
new_ad_data_str = [{k: str(v) for k, v in ad.items()} for ad in new_ad_data]
new_ad_hashed = hasher.transform(new_ad_data_str)

# 2. Predict the probabilities using the *loaded* model
click_probabilities = model.predict_proba(new_ad_hashed)[:, 1]

# 3. Show recommendations
for i, prob in enumerate(click_probabilities):
    print(f"\nAd {i+1}:")
    print(f"Predicted Click Probability: {prob:.4f}")
    if prob > 0.5: # 0.5 is a basic threshold
        print("Recommendation: SHOW AD")
    else:
        print("Recommendation: DO NOT SHOW AD")