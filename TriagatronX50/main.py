from flask import Flask, render_template, request, jsonify, redirect, url_for
import yaml
import app
from flask_mysqldb import MySQL
import nltk
#libraries for text processing
import string 
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pandas as pd
stop_words = set(stopwords.words('english'))


app = Flask(__name__)

def preprocess_reviews(review):
    """
    List of steps to prepare the text derived from the reviews so that it can be processed by the model
    """
    
    #convert text to lowercase
    review = review.lower()
    
    #remove urls. done using regular expression
    review = re.sub(r"http\S+|www\S+|https\S+", "", review, flags=re.MULTILINE)
    
    #remove punctuation
    review = review.translate(str.maketrans("", "", string.punctuation))
    
    #remove any users referenced and remove hashtags, using regular expression
    review = re.sub(r'\@\w+|\#', "", review)
    review = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", review)
    
    
    #remove stopwords
    review_tokens = word_tokenize(review)
    #create list of words for each word in the review tokens. 
    #Only include words that are not in the stopwords!
    filtered_words = [word for word in review_tokens if word not in stop_words]
    
    # #stemming
    # ps = PorterStemmer()
    # stemmed_words = [ps.stem(w) for w in filtered_words]
    
    #lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
    
    return " ".join(lemma_words)

#Configure SQL db
db = yaml.safe_load(open('datab.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)

with app.app_context():
    cur = mysql.connection.cursor()
    # cur.execute("SELECT * FROM journals")
    # df = cur.fetchall()
    query = ("SELECT * FROM journals")
    df = pd.read_sql(query, mysql.connection)
    processed_text = df['processed_text']
    review = df['journal_entry']
    review.fillna("No Text", inplace = True)

    #df1 = df.dropna(how='any',axis=0) 
    # df1 = df[df["text"].notnull()]
    review = review

    def preprocess_reviews(review):
        
        #convert text to lowercase
        review = review.lower()
        
        #remove urls. done using regular expression
        review = re.sub(r"http\S+|www\S+|https\S+", "", review, flags=re.MULTILINE)
        
        #remove punctuation
        review = review.translate(str.maketrans("", "", string.punctuation))
        
        #remove any users referenced and remove hashtags, using regular expression
        review = re.sub(r'\@\w+|\#', "", review)
        review = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", review)
        
        #remove stopwords
        review_tokens = word_tokenize(review)
        #create list of words for each word in the review tokens. 
        #Only include words that are not in the stopwords!
        filtered_words = [word for word in review_tokens if word not in stop_words]
        
        # #stemming
        # ps = PorterStemmer()
        # stemmed_words = [ps.stem(w) for w in filtered_words]
        
        #lemmatizing
        lemmatizer = WordNetLemmatizer()
        lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
        
        return " ".join(lemma_words)
    
    i = 0
    while i <len(review):
        processed_text[i] = preprocess_reviews(review[i])
        i=i+1

    cur.execute("UPDATE journals SET processed_text = %s",[review])
    #commit changes to db
    mysql.connection.commit()

app.app_context()


