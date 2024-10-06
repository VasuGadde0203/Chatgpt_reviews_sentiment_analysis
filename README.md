ChatGPT Reviews Sentiment Analysis

This project implements sentiment analysis on ChatGPT reviews using a Logistic Regression model and presents the results in a user-friendly Streamlit application. 
The goal is to classify reviews as positive, negative, or neutral based on the sentiment expressed in the text.

Table of Contents :- 

1) Project Overview
2) Dataset
3) Technologies Used
4) Installation
5) Usage
6) Model Training
7) Streamlit App
8) Results
9) Future Improvements
10) Acknowledgments

1) Project Overview
This project aims to analyze user reviews of ChatGPT to understand overall sentiment.
Using the Logistic Regression algorithm, the model classifies reviews into three categories: positive, negative, and neutral.
 The results are then displayed using Streamlit for easy access and interaction.

2) Dataset
The dataset used for this project is the chatgpt_reviews dataset, which contains user reviews along with their respective sentiment labels.
Each review is a text entry that expresses a user's opinion about ChatGPT.

  Dataset Structure
    ---> review: The text of the user review.
    ---> sentiment: The sentiment label (positive, negative, neutral).

4) Technologies Used
  ---> Python
  ---> Pandas
  ---> NumPy
  ---> Scikit-learn
  ---> NLTK
  ---> Streamlit
  ---> Matplotlib/Seaborn (for visualizations)

5) Installation
To run this project, you will need to have Python installed on your system. Follow the steps below to set up the project:

Clone the repository:
---> git clone https://github.com/your_username/chatgpt_reviews_sentiment_analysis.git
---> cd chatgpt_reviews_sentiment_analysis

Create a virtual environment (optional but recommended):
---> python -m venv venv
---> source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
---> pip install -r requirements.txt

6) Usage
Run the Streamlit app:
---> streamlit run app.py
Open your web browser and go to http://localhost:8501 to view the app.
You can input a review in the provided text box and click the "Analyze" button to see the sentiment prediction.

7) Model Training
  a) Data Preprocessing
    ---> Text cleaning (removing punctuation, stop words, and applying tokenization).
    ---> Vectorization of text using techniques like TF-IDF or Count Vectorizer.
     
  b) Model Implementation
    ---> The Logistic Regression model is trained using the preprocessed data.
    ---> Hyperparameters are optimized to improve the model's performance.
  
  c) Model Evaluation
    ---> The model's accuracy, precision, recall, and F1-score are evaluated on a separate test dataset.
    ---> Confusion matrix and classification report are generated to visualize the performance.

8) Streamlit App
The Streamlit app provides a user-friendly interface for sentiment analysis. Key features include:
  ---> Input box for user reviews.
  ---> Display of sentiment prediction results.
  ---> Visualization of model performance metrics.

9) Results
  ---> The model's performance metrics are displayed on the Streamlit app, allowing users to assess how well the model classifies the reviews. The visualizations provide insights into the distribution of sentiments in the dataset.

10) Future Improvements
  ---> Experimenting with other machine learning models (e.g., Random Forest, SVM).
  ---> Implementing advanced NLP techniques (e.g., using pre-trained models like BERT).
  ---> Adding more user interaction features in the Streamlit app.

11) Acknowledgments
  ---> Scikit-learn
  ---> Streamlit
  ---> Natural Language Toolkit (NLTK)
