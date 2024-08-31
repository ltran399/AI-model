import pandas as pd
import numpy as np
import joblib
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def get_average_word2vec(review, model, vector_size):
    words = review.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)
# Step 1: Load the saved model
model = joblib.load('relevance_score_xgboost_model.pkl')

# Step 2: Load the saved TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Step 3: Load the test dataset
df = pd.read_csv('combined_reviews_with_refined_relevance_scores_v2.csv')

# Step 4: Handle NaN values in the 'Review' column
df['Review'] = df['Review'].fillna('')

# Step 5: Transform the data using the saved TF-IDF vectorizer
X_tfidf = vectorizer.transform(df['Review'])

# Step 6: Generate Word2Vec features for each review
word2vec_model = Word2Vec.load('word2vec_model_file')  # Load the saved Word2Vec model

X_word2vec = np.array([get_average_word2vec(review, word2vec_model, 100) for review in df['Review']])

# Step 7: Combine TF-IDF features and Word2Vec features
X_combined = hstack([X_tfidf, X_word2vec])

# Step 8: Make predictions with the loaded model
y_pred = model.predict(X_combined)

# Step 9: Evaluate the model with various metrics
mse = mean_squared_error(df['Relevance_Score'], y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(df['Relevance_Score'], y_pred)
r2 = r2_score(df['Relevance_Score'], y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')