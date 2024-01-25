# Import necessary libraries
import spacy
from flask import Flask, request, jsonify

# Load pre-trained text analytics model
model = spacy.load("model-best")

# Create a Flask web application instance and define route for handling prediction requests
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict_method():
    data = request.get_json()
    sentence = data.get('preprocessed_text', None)
    # Check if preprocessed_text is missing or empty
    if not sentence:
        return
    # Make a prediction using the loaded model
    prediction = {"processed_text": sentence,
                  "positive_probability": model(sentence).cats["POSITIVE"]}
    return prediction

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)









