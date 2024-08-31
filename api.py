from flask import Flask, request, jsonify
import joblib
import numpy as np
from gensim.models import Word2Vec
from scipy.sparse import hstack

# Load the model and vectorizer
model = joblib.load('relevance_score_xgboost_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
word2vec_model = Word2Vec.load('word2vec_model_file')

# Function to generate average word vectors
def get_average_word2vec(review, model, vector_size):
    words = review.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the request
    data = request.get_json(force=True)
    review = data['review']
    
    # Transform the review using the TF-IDF vectorizer
    X_tfidf = vectorizer.transform([review])
    
    # Generate Word2Vec features for the review
    X_word2vec = np.array([get_average_word2vec(review, word2vec_model, 100)])
    
    # Combine features
    X_combined = hstack([X_tfidf, X_word2vec])
    
    # Make prediction
    relevance_score = model.predict(X_combined)
    
    # Return the prediction as a JSON response
    return jsonify({'relevance_score': relevance_score[0]})

if __name__ == '__main__':
    app.run(debug=True)