from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')

# Load and preprocess dataset
train_df = pd.read_csv('dataset.csv')

# Function to remove emoticons
def remove_emoticons(text):
    emoticon_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  # emoticons
                                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                  u"\U00002702-\U000027B0"
                                  u"\U000024C2-\U0001F251"
                                  "]+", flags=re.UNICODE)
    return emoticon_pattern.sub(r'', text).replace('\n\n', ' ')

train_df['Deskripsi'] = train_df['Deskripsi'].apply(remove_emoticons)

# Remove stopwords
stop = stopwords.words('indonesian')
train_df['clean'] = train_df['Deskripsi'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Remove symbols
pattern = r'[$-/:-?{-~!"^_`\[\]]'
train_df['clean'] = train_df.clean.str.replace(pattern, '', regex=True)

# Convert text to lowercase
train_df['clean'] = train_df['clean'].str.lower()

# Initialize vectorizer and calculate TF-IDF vectors
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
train_vectors = vectorizer.fit_transform(train_df['clean'])

# Define stemming function
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

train_df['tags'] = train_df['clean'].apply(stem)
vector = vectorizer.transform(train_df['tags'])

# Recommendation function
def recommend_activities(user_preference, penyelenggara, durasi, user_skills, user_past_activities, top_n=3):
    if user_preference.lower() == 'dalam kampus':
        filtered_df = train_df[train_df['Kategori Umum'] == 'Kegiatan Intra Kampus']
        if penyelenggara:
            filtered_df = filtered_df[filtered_df['Penyelenggara'] == penyelenggara]
    else:
        filtered_df = train_df[train_df['Kategori Umum'] == 'Kegiatan Umum']

    # Filter by duration
    if durasi:
        filtered_df = filtered_df[filtered_df['Durasi'] == durasi]

    if filtered_df.empty:
        return []
        
    user_input = user_skills + " " + user_past_activities
    user_input_vector = vectorizer.transform([user_input])
    filtered_vectors = vectorizer.transform(filtered_df['tags'])
    similarity_scores = cosine_similarity(user_input_vector, filtered_vectors).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    recommendations = []
    for index in top_indices:
        recommendations.append({
            'ID': filtered_df.iloc[index]['ID'],
            'Deskripsi': filtered_df.iloc[index]['Deskripsi'],
            'Durasi':filtered_df.iloc[index]['Durasi'],
            'Similarity Score': similarity_scores[index]
        })
    return recommendations

# Define routes
@app.route('/')
def home():
    return render_template('form.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_preference = request.form['preference']
    user_skills = request.form['skills']
    user_past_activities = request.form['past_activities']
    penyelenggara = request.form.get('penyelenggara')
    durasi = request.form.get('durasi')
    recommendations = recommend_activities(user_preference, penyelenggara, durasi, user_skills, user_past_activities)
    return render_template('recommendations.html', recommendations=recommendations)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
