
# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Load Dataset (Replace with your downloaded dataset path)
data = pd.read_csv("twitter_sentiment_data.csv")  # <-- Change this filename accordingly

# Display first few rows
print("Sample Data:\n", data.head())

# Initialize Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment(text):
    score = sid.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply Sentiment Analysis
data['Predicted_Sentiment'] = data['text'].astype(str).apply(get_sentiment)

# Sentiment Distribution
print("\nSentiment Distribution:\n", data['Predicted_Sentiment'].value_counts())

# Plot Sentiment Counts
sns.countplot(data=data, x='Predicted_Sentiment', palette='viridis')
plt.title('Sentiment Distribution')
plt.show()

# Word Cloud for Positive Tweets
positive_text = " ".join(data[data['Predicted_Sentiment'] == 'Positive']['text'])
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(positive_text)
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Sentiment Word Cloud')
plt.show()

# Word Cloud for Negative Tweets
negative_text = " ".join(data[data['Predicted_Sentiment'] == 'Negative']['text'])
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', colormap='Reds').generate(negative_text)
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Sentiment Word Cloud')
plt.show()
