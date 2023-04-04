from flask import Flask, jsonify, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import json

app = Flask(__name__)

# Load the transaction data into a pandas DataFrame
data = pd.read_json('transactions.json')

# Preprocess the transaction descriptions using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['description'])
y = data['category']

# Train a logistic regression model on the entire dataset
model = LogisticRegression()
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict_category():
    # Get the text from the POST request
    text = request.get_json()['description']

    # Preprocess the text using the same vectorizer
    text_vectorized = vectorizer.transform([text])

    # Predict the category using the trained model
    category = model.predict(text_vectorized)[0]

    # Return the predicted category as a JSON response
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True)
