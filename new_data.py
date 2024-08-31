import pandas as pd
import random

# Example words and phrases in Vietnamese
positive_phrases = [
    "Sản phẩm rất tốt", "Dịch vụ tuyệt vời", "Hài lòng với sản phẩm", "Giao hàng nhanh chóng",
    "Chất lượng tốt", "Giá cả hợp lý", "Rất đáng mua", "Nhân viên nhiệt tình"
]

negative_phrases = [
    "Không hài lòng", "Sản phẩm kém chất lượng", "Dịch vụ tệ", "Giao hàng chậm", 
    "Giá quá cao", "Không đáng tiền", "Thất vọng", "Nhân viên thiếu chuyên nghiệp"
]

neutral_phrases = [
    "Sản phẩm bình thường", "Không có gì đặc biệt", "Giá cả vừa phải", "Giao hàng đúng hẹn",
    "Tạm được", "Không tốt cũng không xấu", "Trung bình", "Không có ý kiến"
]

# Function to estimate relevance score
def estimate_relevance_score(review):
    if any(phrase in review for phrase in positive_phrases):
        return random.uniform(75, 100)  # High relevance for positive reviews
    elif any(phrase in review for phrase in neutral_phrases):
        return random.uniform(50, 75)  # Medium relevance for neutral reviews
    elif any(phrase in review for phrase in negative_phrases):
        return random.uniform(0, 50)  # Low relevance for negative reviews
    else:
        return random.uniform(0, 100)  # Random relevance if the review doesn't match any phrases

# Generate 1000 random reviews with relevance scores
reviews = []
relevance_scores = []

for _ in range(1000):
    # Randomly choose a type of review: positive, negative, or neutral
    review_type = random.choice(['positive', 'negative', 'neutral'])
    
    if review_type == 'positive':
        review = random.choice(positive_phrases)
    elif review_type == 'negative':
        review = random.choice(negative_phrases)
    else:
        review = random.choice(neutral_phrases)
    
    # Estimate the relevance score
    relevance_score = estimate_relevance_score(review)
    
    reviews.append(review)
    relevance_scores.append(relevance_score)

# Create a DataFrame
df = pd.DataFrame({'Review': reviews, 'Relevance_Score': relevance_scores})

# Save the DataFrame to a CSV file
df.to_csv('vietnamese_reviews_with_relevance.csv', index=False)

print("Dataset with relevance scores created and saved as 'vietnamese_reviews_with_relevance.csv'")