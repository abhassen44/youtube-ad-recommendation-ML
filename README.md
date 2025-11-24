# YouTube Ad Recommendation

A machine learning system for predicting Click-Through Rate (CTR) of YouTube advertisements to recommend the most effective ads and improve campaign performance.

## Overview

This project implements a CTR prediction model using logistic regression with feature hashing to handle high-cardinality categorical features. The system recommends the most effective ads to display based on predicted click probability.

## Features

- **CTR Prediction**: Predicts click probability for ad impressions
- **Feature Hashing**: Efficiently handles millions of categorical features
- **Web Interface**: Flask-based UI for real-time recommendations
- **Performance Metrics**: Comprehensive analytics dashboard
- **Scalable**: Handles large datasets with online learning

## Dataset

The project uses the Avazu Click-Through Rate Prediction dataset containing:
- **Training Data**: Historical ad impressions with click labels
- **Test Data**: Unlabeled ad impressions for prediction
- **Features**: Device info, site/app details, banner position, timestamps

## Project Structure

```
ml-mini-2/
├── train.py              # Model training script
├── predict.py            # Standalone prediction script
├── recommender.py        # Recommendation engine
├── app.py                # Flask web application
├── templates/
│   └── index.html        # Web interface
├── train.csv             # Training dataset
├── test.csv              # Test dataset
├── ctr_model.joblib      # Trained model (generated)
├── ctr_hasher.joblib     # Feature hasher (generated)
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-mini-2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python train.py
```

This will:
- Load training data
- Hash categorical features
- Train SGD classifier with log loss
- Save model and hasher to disk

### 2. Make Predictions

```bash
python predict.py
```

Predicts click probability for sample ad impressions.

### 3. Run Web Application

```bash
python app.py
```

Access the web interface at `http://localhost:5000`

Features:
- Select number of records to process
- Choose top N recommendations
- View performance metrics
- Analyze CTR distribution

### 4. Use Recommender Module

```python
from recommender import load_model_and_hasher, score_candidates, recommend
import pandas as pd

model, hasher = load_model_and_hasher()
df = pd.read_csv('test.csv', nrows=1000)
scored = score_candidates(df, hasher, model)
top_ads = recommend(scored, top_n=10)
print(top_ads)
```

## Model Details

- **Algorithm**: Stochastic Gradient Descent (SGD) Classifier
- **Loss Function**: Log loss (logistic regression)
- **Feature Engineering**: FeatureHasher with 2^18 features
- **Training**: Online learning with 3 epochs
- **Evaluation**: ROC-AUC and Log Loss

## Performance Metrics

The web interface displays:
- Average CTR across all candidates
- Median and standard deviation
- Top N average CTR
- CTR improvement percentage
- High/Low CTR ad counts

## Key Components

### train.py
Trains the CTR prediction model using SGD classifier with feature hashing.

### recommender.py
Core recommendation engine with functions:
- `load_model_and_hasher()`: Load saved artifacts
- `score_candidates()`: Score ads with click probability
- `recommend()`: Return top N ads by predicted CTR

### app.py
Flask web application providing REST API and UI for recommendations.

### predict.py
Standalone script demonstrating prediction on new ad data.

## Technologies Used

- **Python 3.x**
- **scikit-learn**: Machine learning
- **pandas**: Data manipulation
- **Flask**: Web framework
- **joblib**: Model serialization

## Future Enhancements

- Deep learning models (Neural Networks)
- Real-time streaming predictions
- A/B testing framework
- Feature importance analysis
- Multi-objective optimization (CTR + Revenue)

## License

MIT License

## Contributors

- Abhas Sen

## Acknowledgments

- Avazu for the CTR prediction dataset
- scikit-learn community
