Fake News Detector: Automated News Classification
Project Overview
In an era of rapid information sharing, distinguishing between factual reporting and misinformation is critical. 
This project implements a Machine Learning solution to classify news headlines as either Real or Fake.
Using a Logistic Regression algorithm and Natural Language Processing (NLP) techniques, this tool analyzes the patterns in news titles to provide an instant
authenticity check through a user-friendly web interface
--------------------------------------------------------------------------------
Technologies Used
Language: Python
Data Analysis: Pandas, NumPy
Machine Learning: Scikit-Learn (Logistic Regression)
Vectorization: TF-IDF 
Deployment: Streamlit
Model Persistence: Joblib
--------------------------------------------------------------------------------
Dataset Details
The model was trained using a comprehensive dataset of news articles:-
True News Database (true.csv.zip): Contains verified articles from reputable sources
Fake News Database (fake.csv.zip): Contains articles labeled as misinformation or satire
Features Used: The model specifically focuses on the Title of the news for rapid classification.
--------------------------------------------------------------------------------
Project Workflow
Following a professional AI/ML pipeline, the project was developed in these stages:-
Data Collection: Merging raw CSV files and labeling them (1 for Real, 0 for Fake).
Data Preprocessing: Cleaning the text using a custom clear_title function to remove noise like newlines and lowercase all text.
Feature Extraction: Converting text into numerical data using the TF-IDF Vectorizer.
Model Training: Training a Logistic Regression classifier on 75% of the data.
Evaluation: Testing the model on the remaining 25% of unseen data to ensure high accuracy.
Deployment: Building an interactive UI with Streamlit for real-world testing.
--------------------------------------------------------------------------------
Results
Accuracy Score: The model achieved a high accuracy of approximately 98% on the test set.
Classification Metrics: The system shows strong precision and recall, effectively identifying both real and fake news categories with minimal false positives.
--------------------------------------------------------------------------------
Repository Structure
Fake_News_Detector.ipynb: The full training pipeline and data analysis.
app.py: The Streamlit application script for the web UI.
model.joblib: The saved weights of the trained Logistic Regression model.
vectorizer.joblib: The saved TF-IDF vectorizer.
true.csv.zip & fake.csv.zip: Raw datasets used for training and testing.
