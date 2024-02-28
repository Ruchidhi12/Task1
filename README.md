import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# Load the dataset
df = pd.read_csv('sms_spam_collection.csv')
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)
# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
# TransX_train_tfidf = vectorizer.fit_transform(X_train)
form the training and testing data into TF-IDF vectors
X_test_tfidf = vectorizer.transform(X_test)
# Create a Naive Bayes classifier
classifier = MultinomialNB()
# Train the classifier
classifier.fit(X_train_tfidf, y_train)
# Evaluate the classifier
y_pred = classifier.predict(X_test_tfidf)
# Calculate the accuracy
accuracy = (y_pred == y_test).mean()
# Print the accuracy
print('Accuracy:', accuracy)


