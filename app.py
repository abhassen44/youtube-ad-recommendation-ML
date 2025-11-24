from flask import Flask, render_template, request, jsonify
import pandas as pd
from recommender import load_model_and_hasher, score_candidates, recommend

app = Flask(__name__)
model, hasher = load_model_and_hasher()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    num_records = int(request.form.get('num_records', 1000))
    top_n = int(request.form.get('top_n', 10))
    
    df = pd.read_csv('test.csv', nrows=num_records)
    scored = score_candidates(df, hasher, model)
    recommendations = recommend(scored, top_n)
    
    probs = scored['click_prob']
    metrics = {
        'avg_click_prob': float(probs.mean()),
        'max_click_prob': float(probs.max()),
        'min_click_prob': float(probs.min()),
        'top_avg_click_prob': float(recommendations['click_prob'].mean()),
        'median_click_prob': float(probs.median()),
        'std_click_prob': float(probs.std()),
        'high_ctr_count': int((probs > 0.2).sum()),
        'low_ctr_count': int((probs < 0.05).sum()),
        'improvement': float((recommendations['click_prob'].mean() / probs.mean() - 1) * 100)
    }
    
    return jsonify({
        'recommendations': recommendations.to_dict('records'),
        'total_candidates': len(df),
        'metrics': metrics
    })

if __name__ == '__main__':
    app.run(debug=True)
