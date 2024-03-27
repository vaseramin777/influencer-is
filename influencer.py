import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('influencer_data.csv')

# Preprocess the data
def preprocess_data(df):
    # Remove unnecessary columns
    df = df.drop(['Unnamed: 0', 'id', 'username', 'name', 'profile_picture', 'followers', 'following', 'posts', 'engagement_rate', 'category'], axis=1)
    
    # Convert categorical variables to numerical variables
    df['category'] = pd.Categorical(df['category']).codes
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('category', axis=1), df['category'], test_size=0.2, random_state=42)
    
    # Create a pipeline for preprocessing the data
    preprocessor = make_pipeline(StandardScaler(), PCA(n_components=2))
    
    # Fit the preprocessor to the training data
    preprocessor.fit(X_train)
    
    # Transform the training and testing data
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create a Streamlit app
st.title('Influencerowild')

# User input for the influencer's details
user_input = st.text_input('Enter the influencer\'s details:')

# Preprocess the user input
user_input = user_input.lower().split()

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the user input
vectorizer.fit(user_input)

# Transform the user input into a vector
user_input_vector = vectorizer.transform([user_input])

# Make a prediction on the user input
prediction = clf.predict(user_input_vector)

# Display the prediction
st.write(f'The predicted category for the influencer is: {prediction[0]}')
