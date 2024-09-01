from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

app = Flask(__name__)

# Load the pre-trained models and vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('relevance_score_ridge_model.pkl')

@app.route('/')
def index():
    return "Welcome to the Review Relevance Scoring API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the POST request
        data = request.json
        
        # Ensure 'Review' is in the incoming data
        if 'Review' not in data:
            return jsonify({'error': 'Review field is required'}), 400
        
        # Extract the review text
        review_text = data['Review']
        
        # Transform the review text using the TF-IDF vectorizer
        X_tfidf = vectorizer.transform([review_text])
        
        # Make the prediction using the Ridge regression model
        prediction = model.predict(X_tfidf)
        
        # Return the prediction result
        return jsonify({'Relevance_Score': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)