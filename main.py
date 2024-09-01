import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import joblib
from scipy.sparse import hstack, csr_matrix

# Step 1: Load the dataset
df = pd.read_csv('combined_reviews_with_refined_relevance_scores_v2.csv')

# Ensure that the 'Review' column exists
if 'Review' not in df.columns:
    raise ValueError("The DataFrame does not contain a 'Review' column.")

# Step 2: Downsample the data (if necessary)
df_sample = df.sample(frac=0.3, random_state=42)  # Further reduce to 30% of the data

# Step 3: Handle NaN values in the 'Review' column
df_sample['Review'] = df_sample['Review'].fillna('')  # Replace NaN with empty string

# Step 4: TF-IDF Vectorization
# Further reduce the number of features to lower memory usage
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 1))  # Use unigrams only
X_tfidf = vectorizer.fit_transform(df_sample['Review'])

# Save the TF-IDF vectorizer for later use
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Step 5: Define target variable
y = df_sample['Relevance_Score']

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 7: Initialize and Train a Ridge Regression Model
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test)

# Step 9: Save the model and evaluate it
joblib.dump(model, 'relevance_score_ridge_model.pkl')

mse = mean_squared_error(y_test, y_pred)
print(f'Ridge Regression Mean Squared Error: {mse}')