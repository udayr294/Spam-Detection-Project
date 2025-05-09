import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
data = pd.read_csv("/Users/udaykumar/Desktop/spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['not_spam','spam'])

# Split data
X = data['Message']
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Vectorize text
# cv = CountVectorizer(stop_words='english')
cv = TfidfVectorizer(stop_words='english')
X_train_vec = cv.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Streamlit UI
st.title("Spam Detection")
input_mess = st.text_input("Enter message here")

if st.button("Predict"):
    input_data = cv.transform([input_mess])
    prediction = model.predict(input_data)
    st.write("Prediction:", prediction[0])
    
