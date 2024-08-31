import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from gensim.models import Word2Vec
import numpy as np
from scipy.sparse import hstack
import joblib

# Define the get_average_word2vec function
def get_average_word2vec(review, model, vector_size):
    words = review.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Step 1: Load the dataset
df = pd.read_csv('vietnamese_reviews_with_relevance.csv')

# Step 2: Handle NaN values in the 'Review' column
df['Review'] = df['Review'].fillna('')

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
X_tfidf = vectorizer.fit_transform(df['Review'])
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Step 4: Word Embeddings with Word2Vec
tokenized_reviews = [review.split() for review in df['Review']]
word2vec_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save('word2vec_model_file')
X_word2vec = np.array([get_average_word2vec(review, word2vec_model, 100) for review in df['Review']])

# Step 5: Combine TF-IDF features and Word2Vec features
X_combined = hstack([X_tfidf, X_word2vec])

# Step 6: Define target variable
y = df['Relevance_Score']

# Step 7: Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 8: Initialize and train the XGBoost Regressor on the training set
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 9: Validate the model on the validation set
y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f'Validation MSE: {val_mse}')

# Step 10: Test the model on the test set
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f'Test MSE: {test_mse}')
