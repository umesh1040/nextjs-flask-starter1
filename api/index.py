from flask import Flask, request, jsonify
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
import joblib
import numpy as np
 
from flask_cors import CORS  # Import CORS from flask_cors

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

tfidf_vectorizer = TfidfVectorizer()
fitted_tfidf_vectorizer = None

def identify_dataset(user_input):
    programming_languages = ["cpp", "python", "c", "java", "javascript", "php"]
    dataset_paths = {
        "cpp": "cpp_dataset.csv",
        "python": "python_dataset.csv",
        "c": "c_dataset.csv",
        "java": "java_dataset.csv",
        "javascript": "javascript_dataset.csv",
        "php": "php_dataset.csv",
    }

    for lang in programming_languages:
        if lang in user_input.lower():
            return dataset_paths.get(lang)

    return None

def process_dataset(dataset_path):
    global fitted_tfidf_vectorizer

    dataset = pd.read_csv(dataset_path, encoding="ISO-8859-1")

    processed_questions = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    for question in dataset["Question"]:
        tokens = word_tokenize(question)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        processed_questions.append(" ".join(lemmatized_tokens))

    global fitted_tfidf_vectorizer
    if fitted_tfidf_vectorizer is None:
        fitted_tfidf_vectorizer = tfidf_vectorizer.fit(processed_questions)
        joblib.dump(fitted_tfidf_vectorizer, "fitted_tfidf_vectorizer.pkl")

    fitted_tfidf_vectorizer = joblib.load("fitted_tfidf_vectorizer.pkl")

    tfidf_matrix = fitted_tfidf_vectorizer.transform(processed_questions)

    return tfidf_matrix

def process_user_input(user_input):
    spell = SpellChecker()
    misspelled = spell.unknown(user_input.split())
    corrected_input = " ".join(
        spell.correction(word) if word in misspelled else word
        for word in user_input.split()
    )

    tokens = word_tokenize(corrected_input)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    global fitted_tfidf_vectorizer
    if lemmatized_tokens and fitted_tfidf_vectorizer:
        tfidf_matrix = fitted_tfidf_vectorizer.transform([" ".join(lemmatized_tokens)])
        return tfidf_matrix
    else:
        print("User input could not be processed.")
        return None

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.json
    print("User input-------------", data)
    user_input = data['userInput']
    dataset_path = identify_dataset(user_input)

    if dataset_path:
        dataset_tfidf_matrix = process_dataset(dataset_path)
        joblib.dump(dataset_tfidf_matrix, "dataset_tfidf_matrix.pkl")
    else:
        results=[]
        result = {  # Convert to regular integer
                "answer": "No suitable answer in the dataset"
            }
        results.append(result)
        return results

    user_tfidf_matrix = process_user_input(user_input)

    dataset_tfidf_matrix = joblib.load("dataset_tfidf_matrix.pkl")

    similarity_scores = cosine_similarity(user_tfidf_matrix, dataset_tfidf_matrix)
    top_indices = np.argsort(similarity_scores[0])[-1:][::-1]

    dataset = pd.read_csv(dataset_path, encoding="ISO-8859-1")
    results = []
    for idx in top_indices:
        if similarity_scores[0][idx] > 0.5:
            result = {  # Convert to regular integer
                "answer": dataset['Answer'][idx]
            }
            results.append(result)

    if not results:
        result = {  # Convert to regular integer
                "answer": "No suitable answer in the dataset"
            }
        results.append(result)
         

    return jsonify(results), 200
if __name__ == '__main__':
    app.run(debug=True)


@app.route("/api/python", methods=['GET'])
def hello_world():
    return "<p>Hello, World!</p>"


