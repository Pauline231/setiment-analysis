from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the saved model
MODEL_PATH = 'model/model.keras'  # Replace with your model's path
model = tf.keras.models.load_model(MODEL_PATH)

def load_dict(filename):
  with open(filename, "r") as json_file:
    loaded_data = json.load(json_file)
  return loaded_data

word_to_idx = load_dict('dicts/word_to_idx.json')

max_len_dict = load_dict('dicts/max_length.json')
MAX_LENGTH = max_len_dict["max_length"]

idx_to_label = {
  0: 1,
  1: 2,
  2: 3,
  3: 4,
  4: 5
}
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(caption):
  caption = caption.lower()
  caption = re.sub("[^a-z]+"," ",caption) # removing puncuation using regex
  caption = caption.split()
  caption = [word for word in caption if len(word) > 1] # removing words less than 1

  # removing stop words
  caption = [word for word in caption if word not in stop_words]

  # lemmattization
  caption = [lemmatizer.lemmatize(word) for word in caption]

  caption = " ".join(caption)
  caption = "startseq "+caption+" endseq"
  return caption

def convert_word_to_token(sentence, max_length):
  words = sentence.split()
  word_tokens = []

  # mapping to token
  for word in words:
    if word in word_to_idx:
      word_tokens.append(word_to_idx[word])
    else:
      word_tokens.append(word_to_idx['[UNK]'])

  # padding with 0 if its lenght is less than max lenght
  return tf.keras.utils.pad_sequences([word_tokens], maxlen=max_length, padding='post', value=0)

def predict_rating(user_input):
  print(f"Review: {user_input}")
  clean_input = clean_text(user_input)
  seq = convert_word_to_token(clean_input, MAX_LENGTH)
  y_pred = model.predict(seq)
  pred_calss = idx_to_label[np.argmax(y_pred)]
  print(f"Predicted rating for your review is: {pred_calss}")
  return pred_calss

#predict_rating("The managers were so unfriendly. I dont recommend to anyone.")


@app.route("/")
def hello_world():
 return "<p>Hello world. the server is working.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.form['input']  # Expecting JSON with 'inputs' key
        val = data # Convert inputs to NumPy array
        
        res = predict_rating(val)
        return jsonify({'predictions': res})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
