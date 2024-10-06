import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
nltk.download('stopwords')
# Load the stopwords and stemmer
stop_words = set(stopwords.words('english'))
port_stem = PorterStemmer()

# Function for text preprocessing
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Load the pickled model and vectorizer
def load_model():
    with open('sentiment_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit UI
st.title('ChatGPT Review Sentiment Analysis')

# Text input for user review
user_input = st.text_area('Enter your review below', '')

# Predict sentiment when the user submits the review
if st.button('Predict Sentiment'):
    if user_input:
        # Preprocess the input
        cleaned_input = stemming(user_input)

        # Vectorize the input
        input_vectorized = vectorizer.transform([cleaned_input]).toarray()

        # Predict the sentiment
        prediction_proba = model.predict_proba(input_vectorized)[0]
        predicted_label = model.predict(input_vectorized)[0]

        # Calculate percentages
        negative_percentage = prediction_proba[0] * 100
        positive_percentage = prediction_proba[1] * 100

        # Display results
        st.write(f"### Prediction: {'Positive' if predicted_label == 1 else 'Negative'}")
        st.write(f"#### Positive Probability: {positive_percentage:.2f}%")
        st.write(f"#### Negative Probability: {negative_percentage:.2f}%")

        # Display a pie chart for probabilities
        labels = ['Positive', 'Negative']
        probabilities = [positive_percentage, negative_percentage]
        # fig, ax = plt.subplots()
        # ax.pie(probabilities, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
        # st.pyplot(fig)

        fig = go.Figure(data=[go.Pie(labels=labels, values=probabilities, hole=0.3, 
                                     marker=dict(colors=['green', 'red']))])
        fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=16)

        # Display the pie chart
        st.plotly_chart(fig)
    else:
        st.write('Please enter a review to analyze.')
