import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
df = pd.read_csv("Twitter_Data.csv")

# 2. Rename columns for simplicity 
df = df.rename(columns={'clean_text': 'text', 'category': 'sentiment'})

# 3. Drop any missing rows
df = df.dropna(subset=['text', 'sentiment'])

# 4. Convert sentiment column to integer
df['sentiment'] = df['sentiment'].astype(int)

# 5. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# 6. Convert text to numeric features using CountVectorizer (Bag of Words)
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train a simple Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 8. Make predictions
y_pred = model.predict(X_test_vec)

# 9. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10.Plot sentiment distribution
df['sentiment'].value_counts().sort_index().plot(kind='bar', title='Sentiment Distribution', xlabel='Sentiment', ylabel='Count')
plt.xticks(ticks=[0, 1, 2], labels=['Negative', 'Neutral', 'Positive'], rotation=0)
plt.show()
