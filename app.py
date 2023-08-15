import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    # Load the trained TfidfVectorizer
    with open('tfidf.pkl', 'rb') as tfidf_file:
        tfidfd = pickle.load(tfidf_file)

    # Load the trained classifier
    with open('clf.pkl', 'rb') as clf_file:
        clf = pickle.load(clf_file)

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        
        # Load the trained TfidfVectorizer
        with open('tfidf.pkl', 'rb') as tfidf_file:
            tfidfd = pickle.load(tfidf_file)
        
        # Load the trained classifier
        with open('clf.pkl', 'rb') as clf_file:
            clf = pickle.load(clf_file)

        df = pd.read_csv("UpdatedResumeDataset.csv")
        tfidfd.fit(df['Resume'])
        
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        category_mapping = {
            15: "Java Developer",
            # ... (other mappings)
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()
