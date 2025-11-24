# Project Overview: Effective Targeting of Advertisements

## Problem Statement

Online advertising platforms need to maximize Click-Through Rate (CTR) by showing the right ads to the right users. This project builds a machine learning system to predict which advertisements are most likely to be clicked, enabling better ad targeting and higher campaign ROI.

## Business Impact

- **Increased Revenue**: Higher CTR leads to more conversions
- **Better User Experience**: Users see more relevant ads
- **Optimized Ad Spend**: Advertisers get better value for money
- **Data-Driven Decisions**: Replace guesswork with ML predictions

## Technical Approach

### 1. Data Processing
- Handle high-cardinality categorical features (millions of unique values)
- Use Feature Hashing to create fixed-size feature vectors
- Convert all features to string format for consistent hashing

### 2. Model Selection
- **Algorithm**: SGD Classifier with log loss
- **Why SGD?**: 
  - Handles large datasets efficiently
  - Online learning capability
  - Memory efficient
  - Fast training and prediction

### 3. Feature Engineering
- **Feature Hashing**: Maps categorical values to 2^18 dimensional space
- **Benefits**:
  - No need to store vocabulary
  - Handles unseen categories
  - Fixed memory footprint
  - Fast transformation

### 4. Model Training
- **Loss Function**: Log loss (for probability calibration)
- **Epochs**: 3 passes over data
- **Evaluation Metrics**: ROC-AUC and Log Loss
- **Output**: Calibrated click probabilities (0-1)

### 5. Recommendation System
- Score all candidate ads with click probability
- Rank by predicted CTR
- Return top N recommendations
- Calculate performance metrics

## System Architecture

```
┌─────────────┐
│  train.csv  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Feature Hash   │
│  + SGD Train    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐     ┌──────────────┐
│  Model Files    │────▶│  Flask App   │
│  (.joblib)      │     └──────┬───────┘
└─────────────────┘            │
                               ▼
                        ┌──────────────┐
                        │  Web UI      │
                        │  (Browser)   │
                        └──────────────┘
```

## Key Metrics

### Model Performance
- **ROC-AUC**: Measures ranking quality (higher is better)
- **Log Loss**: Measures probability calibration (lower is better)

### Business Metrics
- **Average CTR**: Overall click probability
- **Top N CTR**: Average CTR of recommended ads
- **CTR Improvement**: % increase from targeting
- **High/Low CTR Counts**: Distribution analysis

## Files Description

| File | Purpose |
|------|---------|
| `train.py` | Train the CTR prediction model |
| `predict.py` | Standalone prediction script |
| `recommender.py` | Core recommendation engine |
| `app.py` | Flask web application |
| `EDA.py` | Exploratory data analysis |
| `templates/index.html` | Web interface |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `SETUP.md` | Setup instructions |
| `run.bat` | Windows launcher script |

## Usage Scenarios

### Scenario 1: Batch Recommendations
```python
# Load model
model, hasher = load_model_and_hasher()

# Score candidates
df = pd.read_csv('test.csv', nrows=10000)
scored = score_candidates(df, hasher, model)

# Get top ads
top_ads = recommend(scored, top_n=100)
```

### Scenario 2: Real-time Prediction
```python
# Single ad impression
ad_data = {
    'hour': '14102103',
    'C1': '1005',
    'banner_pos': '1',
    # ... other features
}

# Transform and predict
X = hasher.transform([ad_data])
click_prob = model.predict_proba(X)[0, 1]
```

### Scenario 3: Web Interface
1. Open browser to `http://localhost:5000`
2. Select number of records to process
3. Choose top N recommendations
4. View results and metrics

## Performance Optimization

### Current Implementation
- Feature hashing: O(n) time, O(1) space
- SGD training: O(n) time per epoch
- Prediction: O(n) time for n samples

### Potential Improvements
1. **Parallel Processing**: Use joblib for multi-core scoring
2. **Batch Prediction**: Process ads in batches
3. **Caching**: Cache frequent feature combinations
4. **Model Compression**: Reduce model size for faster loading

## Future Enhancements

### Short-term
- [ ] Add more features (C14-C21, device_conn_type)
- [ ] Implement cross-validation
- [ ] Add model versioning
- [ ] Create API endpoints

### Medium-term
- [ ] Try ensemble models (Random Forest, XGBoost)
- [ ] Implement A/B testing framework
- [ ] Add real-time streaming predictions
- [ ] Build monitoring dashboard

### Long-term
- [ ] Deep learning models (Neural Networks)
- [ ] Multi-objective optimization (CTR + Revenue)
- [ ] Personalization engine
- [ ] Auto-ML pipeline

## Lessons Learned

1. **Feature Hashing**: Effective for high-cardinality categorical
2. **Online Learning**: SGD works well for large datasets
3. **Probability Calibration**: Log loss ensures reliable probabilities
4. **Simplicity**: Simple models can be very effective

## References

- Avazu Click-Through Rate Prediction Dataset
- scikit-learn Documentation
- Feature Hashing for Large Scale Multitask Learning (Weinberger et al.)
- Online Learning and Stochastic Approximations (Bottou)

## Contact

For questions or contributions, please open an issue or submit a pull request.

---

