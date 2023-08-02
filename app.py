import  streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')

ps = PorterStemmer()


def transform_text(text): 
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    
    for i in text: 
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

tfidf = pickle.load(open('vectorizer4.pkl', 'rb'))
model  = pickle.load(open('model4.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the SMS or the email")

if st.button('Predict'):

    #1 preprocess
    transformed_sms = transform_text(input_sms)
    #2 Vectorize
    vector_input = tfidf.transform([transformed_sms])
    #3 Predict
    result = model.predict(vector_input)[0]
    #4 Display

    if result == 1:
        st.write("The message is spam!")
    elif result == 0:
        st.write("The message isn't spam.")


