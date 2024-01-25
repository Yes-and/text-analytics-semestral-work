#!/usr/bin/env python
# coding: utf-8

# In[74]:


import spacy
import re
from spacy.language import Language
from spacy.tokens import Doc
from spacy.language import Language
from langdetect import detect, DetectorFactory, LangDetectException
from spacy.tokens import Doc
from spacy.language import Language
import requests


# In[43]:


nlp = spacy.load('en_core_web_sm')


# In[44]:


@Language.component("filter_length")
def filter_length(doc):
    # Count the number of word tokens in the document
    word_count = len([token for token in doc if token.is_alpha])

    # Check if the document has more than 1 word
    if word_count > 1:
        return doc
    else:
        # Return an empty Doc if the condition is not met
        return Doc(doc.vocab, words=[])

# Add the component to the pipeline
nlp.add_pipe("filter_length", name="length_filter", first=True)


# In[45]:


@Language.component("filter_emojis")
def filter_emojis(doc):
    # Check if the document is empty (as a result of the previous filter)
    if len(doc) == 0:
        return doc

    # Regular expression pattern to match emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    # Remove emojis from the text
    new_text = emoji_pattern.sub(r'', doc.text)

    # Create a new Doc with the emoji-free text
    return Doc(doc.vocab, words=new_text.split())

# Add the component to the pipeline
nlp.add_pipe("filter_emojis", name="emoji_filter", after="length_filter")


# In[46]:


# Set seed for langdetect to get deterministic results
@Language.component("filter_language")
def filter_language(doc):
    # Check if the document is empty
    if len(doc) == 0:
        return doc

    try:
        # Detect the language of the text
        if detect(doc.text) == 'en':
            return doc
        else:
            # Return an empty Doc if the text is not English
            return Doc(doc.vocab, words=[])
    except LangDetectException:
        # Handle exception if language detection fails
        return Doc(doc.vocab, words=[])

# Add the component to the pipeline
nlp.add_pipe("filter_language", name="filter_language", after="emoji_filter")


# In[47]:


# Custom component for keeping only standard English sentence symbols
@Language.component("filter_english_symbols")
def filter_english_symbols(doc):
    # Define a regular expression pattern to match all types of punctuation
    punctuation_pattern = re.compile(r'[^\w\s]+')

    # Use re.sub() to replace punctuation with an empty space in the entire text
    processed_text = re.sub(punctuation_pattern, ' ', doc.text)

    # Create a new Doc with the processed text
    new_doc = Doc(doc.vocab, words=processed_text.split())
    return new_doc

# Add the component to the pipeline
nlp.add_pipe("filter_english_symbols", name="filter_english_symbols", after="filter_language")


# In[48]:


@Language.component("remove_stopwords")
def remove_stopwords(doc):
    # Check if the document is empty
    if len(doc) == 0:
        return doc

    # Remove stop words from the text
    stop_words = set(token.text for token in doc if token.is_stop and token.text.lower() not in {"not", "do", "can","don"})
    filtered_tokens = [token.text for token in doc if token.text not in stop_words]
    # Check if filtered tokens are empty
    if not filtered_tokens:
        return doc

    # Create a new Doc with the text without stop words
    new_doc = Doc(doc.vocab, words=filtered_tokens)
    return new_doc

# Add the component to the pipeline
nlp.add_pipe("remove_stopwords", name="remove_stopwords", after="filter_english_symbols")


# In[49]:


contractions = {
    "t": "not",
    "don": "do",
    "doesn": "does",
    "didn": "did",
    "won": "will",
    "cant": "cannot",
}

@Language.component("expand_contractions")
def expand_contractions(doc):
    # Check if the document is empty
    if len(doc) == 0:
        return doc

    # Expand contractions in the text
    expanded_tokens = [contractions.get(token.text, token.text) for token in doc]

    # Check if expanded tokens are empty
    if not expanded_tokens:
        return doc

    # Create a new Doc with the expanded text
    new_doc = Doc(doc.vocab, words=expanded_tokens)
    return new_doc

# Add the component to the pipeline
nlp.add_pipe("expand_contractions", name="expand_contractions", after="remove_stopwords")


# In[50]:


@Language.component("preprocess_text")
def preprocess_text(doc):
    # Check if the document is empty
    if len(doc) == 0:
        return doc

    # Process the text: lowercase, remove punctuation, and lemmatize
    lemmatized_tokens = [token.lemma_.lower() for token in doc if token.is_alpha]

    if not lemmatized_tokens:
      return doc

    # Create a new Doc with the processed text
    # Note: This method preserves the original Doc's properties
    new_doc = Doc(doc.vocab, words=lemmatized_tokens)
    return new_doc

# Add the component to the pipeline
nlp.add_pipe("preprocess_text", name="preprocess_text", last=True)


# In[ ]:


from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route('/process_and_predict', methods=['POST'])
def process_and_predict():
    data = request.get_json()
    sentence = data.get('text_to_process') 
    doc = nlp(sentence)
    preprocessed_text = doc.text
    to_process = {"preprocessed_text":preprocessed_text}
    response = requests.post("http://prediction-service:5001/predict", json=to_process)
    return response.json()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

