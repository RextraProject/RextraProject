# ini kayanya lebih oke tpi masih harus diimprove lagi

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')
nltk.download('wordnet')

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

# Apply emoticon removal
train_df['Deskripsi'] = train_df['Deskripsi'].apply(remove_emoticons)
train_df['Persyaratan Pendaftaran'] = train_df['Persyaratan Pendaftaran'].apply(remove_emoticons)
train_df['Skill yang didapat'] = train_df['Skill yang didapat'].apply(remove_emoticons)

# Remove stopwords
stop = stopwords.words('indonesian')
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop])

train_df['Deskripsi'] = train_df['Deskripsi'].apply(remove_stopwords)
train_df['Persyaratan Pendaftaran'] = train_df['Persyaratan Pendaftaran'].apply(remove_stopwords)
train_df['Skill yang didapat'] = train_df['Skill yang didapat'].apply(remove_stopwords)

# Remove symbols and convert to lowercase
pattern = r'[$-/:-?{-~!"^_`\[\]]'
train_df['Deskripsi'] = train_df['Deskripsi'].str.replace(pattern, '', regex=True).str.lower()
train_df['Persyaratan Pendaftaran'] = train_df['Persyaratan Pendaftaran'].str.replace(pattern, '', regex=True).str.lower()
train_df['Skill yang didapat'] = train_df['Skill yang didapat'].str.replace(pattern, '', regex=True).str.lower()

# Combine relevant columns into one
train_df['combined'] = train_df['Deskripsi'] + ' ' + train_df['Persyaratan Pendaftaran'] + ' ' + train_df['Skill yang didapat']

# Define lemmatization function
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

train_df['combined'] = train_df['combined'].apply(lemmatize)

# Initialize vectorizer and calculate TF-IDF vectors
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
train_vectors = vectorizer.fit_transform(train_df['combined'])

# Recommendation function
def recommend_activities(user_input, user_preference, penyelenggara, durasi, top_n=3):
    # Filter by user preference
    if user_preference.lower() == 'dalam kampus':
        filtered_df = train_df[train_df['Kategori Umum'] == 'Kegiatan Intra Kampus']
        if penyelenggara:
            filtered_df = filtered_df[filtered_df['Penyelenggara'] == penyelenggara]
    else:
        filtered_df = train_df[train_df['Kategori Umum'] == 'Kegiatan Umum']

    # Filter by duration
    if durasi:
        filtered_df = filtered_df[filtered_df['Durasi'] == durasi]

    # Check if filtered dataframe is empty
    if filtered_df.empty:
        return []

    # Combine filtered data into one string for vectorization
    filtered_combined = filtered_df['combined']

    # Vectorize the filtered combined column
    filtered_vectors = vectorizer.transform(filtered_combined)

    # Calculate similarity scores between user input and filtered vectors
    user_input_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_input_vector, filtered_vectors).flatten()

    # # Get the top N recommendations
    # top_indices = similarity_scores.argsort()[-top_n:][::-1]

    # Get the top N recommendations with similarity score above threshold
    top_indices = [index for index in similarity_scores.argsort()[::-1] if similarity_scores[index] > 0.1][:top_n]

    recommendations = []
    for index in top_indices:
        recommendations.append({
            'ID': filtered_df.iloc[index]['ID'],
            'Deskripsi': filtered_df.iloc[index]['Deskripsi'],
            'Durasi': filtered_df.iloc[index]['Durasi'],
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

    # Combine user inputs
    user_input = f"{user_preference} {user_skills} {user_past_activities} {penyelenggara} {durasi}"
    recommendations = recommend_activities(user_input, user_preference, penyelenggara, durasi)
    
    return render_template('recommendations.html', recommendations=recommendations)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
