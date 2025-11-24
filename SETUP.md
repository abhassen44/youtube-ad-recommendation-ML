# Setup Guide

## Quick Start

Follow these steps to get the project running:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Data Files

Ensure you have these CSV files in the project directory:
- `train.csv` - Training dataset with click labels
- `test.csv` - Test dataset for predictions

### 3. Train the Model

```bash
python train.py
```

This will:
- Load and process training data
- Hash categorical features
- Train SGD classifier
- Save `ctr_model.joblib` and `ctr_hasher.joblib`

Expected output:
```
Loading training data...
Loaded XXXXX training samples
Preparing features...
Hashing features...
Training model...
Epoch 1: AUC=0.XXXX, LogLoss=0.XXXX
Epoch 2: AUC=0.XXXX, LogLoss=0.XXXX
Epoch 3: AUC=0.XXXX, LogLoss=0.XXXX
Saving model...
Model saved successfully!
```

### 4. Run the Web Application

```bash
python app.py
```

Open your browser and navigate to: `http://localhost:5000`

### 5. (Optional) Explore Data

```bash
python EDA.py
```

### 6. (Optional) Test Predictions

```bash
python predict.py
```

## Troubleshooting

### Issue: "Model files not found"
**Solution**: Run `python train.py` first to generate the model files.

### Issue: "CSV file not found"
**Solution**: Ensure `train.csv` and `test.csv` are in the project root directory.

### Issue: "Module not found"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: Port 5000 already in use
**Solution**: Modify `app.py` and change the port:
```python
app.run(debug=True, port=5001)
```

## Project Workflow

```
1. Data Collection → train.csv, test.csv
2. Training → python train.py → ctr_model.joblib, ctr_hasher.joblib
3. Deployment → python app.py → Web Interface
4. Prediction → Score ads and recommend top performers
```

## Features Explained

### Categorical Features Used:
- `hour`: Timestamp of ad impression
- `C1`: Anonymized categorical variable
- `banner_pos`: Position of banner ad
- `site_id`: Site identifier
- `site_domain`: Site domain hash
- `site_category`: Site category
- `app_id`: App identifier
- `app_domain`: App domain hash
- `app_category`: App category
- `device_id`: Device identifier
- `device_ip`: IP address hash
- `device_model`: Device model
- `device_type`: Device type (mobile/tablet/desktop)

### Model Architecture:
- **Feature Hashing**: Converts high-cardinality categoricals to fixed-size vectors
- **SGD Classifier**: Online learning algorithm for large datasets
- **Log Loss**: Optimizes for probability calibration
- **3 Epochs**: Multiple passes over data for better convergence

## Next Steps

1. Experiment with different model parameters
2. Add more features (device_conn_type, C14-C21)
3. Try different algorithms (Random Forest, XGBoost)
4. Implement cross-validation
5. Deploy to production server

## Support

For issues or questions, please refer to the README.md or create an issue in the repository.
