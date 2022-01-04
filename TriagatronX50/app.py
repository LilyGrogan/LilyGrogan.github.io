from flask import Flask, render_template, request, jsonify, redirect, url_for
#To connect the database 
from flask_mysqldb import MySQL
#import pyyaml
import yaml
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
#import scipy
import gensim
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from gensim import corpora, models, similarities
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis.gensim_models
import pyLDAvis
import collections
from collections import defaultdict
from gensim.corpora import Dictionary
# from flask_navigation import Navigation
# from flask_navigation import *
import re
from gensim.test.utils import datapath
import app
import sys
import pickle
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
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
#creating english stopwords
stop_words = set(stopwords.words('english'))
# import gensim, spacy, logging, warnings
# import gensim.corpora as corpora
# from gensim.utils import lemmatize, simple_preprocess
# from gensim.models import CoherenceModel
# import matplotlib.pyplot as plt5

app = Flask(__name__)

import os.path
prog = ("/Users/jeram/Documents/NCI/Semester3/TriagatronX50")
directory = os.path.dirname(os.path.abspath(prog))
print(os.path.join(directory, 'datab.yaml'))

#Configure SQL db
db = yaml.safe_load(open('datab.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

#instantiate an object for mysql module (app is the parameter)
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
        review = re.sub(r"\@\'w+|\#", "", review)
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
        processed_text = preprocess_reviews(review[i])
        i=i+1
        cur.execute("UPDATE journals SET processed_text = %s WHERE journal_id = %s",[processed_text, i])

    #cur.execute("UPDATE journals SET processed_text = %s",[review])
    #commit changes to db
    mysql.connection.commit()

app.app_context()


sentiment_model= MultinomialNB()
sentiment_model=pickle.load(open('sentiment_model.pkl', 'rb'))
vec = pickle.load(open("vectorizer.pickle", "rb"))
#preprocess_text = pickle.load(open("preprocess_text.pkl", "rb"))

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    #Post means info should be stored in database
    #Get means info should be displayed directly to user
        if request.method == 'POST':

            details = request.form
            username = details['username']
            passwords = details['passwords']

            # flag to check errors
            error = None
            # check if the 'reset password' form is filled and all parameters are posted
            if not username:
                error = "Username is required."
            elif not passwords:
                error = "Password is required."
            
            if error is None:
                
                #check if account exists in database
                cursor = mysql.connection.cursor()
                cursor.execute( "SELECT * FROM log_on WHERE username = %s AND passwords = %s", (username, passwords))
                checkMatch = cursor.fetchall()  # to get the first record
                #db_pwd = checkMatch[0]

                if checkMatch:
                    return render_template("journal.html")
                else:
                    return render_template("login.html", info="Login Failure")
        else:
            return render_template("login.html")

#Import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import urllib, base64

common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
lda = gensim.models.LdaModel(common_corpus, num_topics=5, alpha='auto', eval_every=5)

# Save model to disk.
temp_file = datapath("model")
lda.save(temp_file)

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/Users/jeram/Documents/NCI/Semester3/TriagatronX50/templates'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with app.app_context():
            mysql.connection.cursor()
            query = ("SELECT * FROM journals")
            df = pd.read_sql(query, mysql.connection)
            df.fillna("No Text", inplace = True)
            df['processed_text'] = df.apply(lambda row: nltk.word_tokenize(row['processed_text']), axis=1)
            clean_tokens = df['processed_text']
            id2word = corpora.Dictionary(clean_tokens)
            corpus = [id2word.doc2bow(text) for text in clean_tokens]
            lda_model = gensim.models.LdaModel(corpus ,id2word=id2word, num_topics=6, alpha='auto', eval_every=5, random_state=300,update_every=1,chunksize=10,passes=10,iterations=300,per_word_topics=True)
            p = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
            pyLDAvis.save_html(p, 'LDA_Visualisation.html')

@app.route('/insight', methods=['GET', 'POST'])
def insight():
    if request.method == 'POST':
        #if request.form.get('action1') == 'See List':
        if request.form.get('submit1') == 'Main Themes':
            mysql.connection.cursor()
            query = ("SELECT * FROM journals")
            df = pd.read_sql(query, mysql.connection) 
            df['processed_text'] = df.apply(lambda row: nltk.word_tokenize(row['processed_text']), axis=1)
            clean_tokens = df['processed_text']
            id2word = corpora.Dictionary(clean_tokens)
            corpus = [id2word.doc2bow(text) for text in clean_tokens]
            lda_model = gensim.models.LdaModel(corpus ,id2word=id2word, num_topics=6, alpha='auto', eval_every=5, random_state=300,update_every=1,chunksize=10,passes=10,iterations=300,per_word_topics=True)

            def create_topic_table(ldamodel=None, corpus=corpus, texts=clean_tokens):
                # Initialise the output
                new_topic_df = pd.DataFrame()

                # Get main topic from each document
                for i, row_list in enumerate(ldamodel[corpus]):
                    row = row_list[0] if ldamodel.per_word_topics else row_list            
                    # print the row for each row
                    row = sorted(row, key=lambda x: (x[1]), reverse=True)
                    # Get the main topic, percentage contribution and the main keywords per document
                    for j, (topic_number, prop_topic) in enumerate(row):
                        if j == 0:  # this will give us the dom topic
                            wordpath = ldamodel.show_topic(topic_number)
                            keywords_topic = ", ".join([word for word, prop in wordpath])
                            new_topic_df = new_topic_df.append(pd.Series([int(topic_number), round(prop_topic,4), keywords_topic]), ignore_index=True)
                        else:
                            break
                new_topic_df.columns = ['Dom_Topic', 'Perc_Contribution', 'Keywords_Topic']

                # Append the review to the end of the output
                contents = pd.Series(clean_tokens)
                new_topic_df = pd.concat([new_topic_df, contents], axis=1)
                return(new_topic_df)

            df_topic_sents_keywords = create_topic_table(ldamodel=lda_model, corpus=corpus, texts=clean_tokens)

            df_dom_topic = df_topic_sents_keywords.reset_index()
            df_dom_topic.columns = ['Doc_No', 'Dominant_Topic', 'Topic_Contribution', 'Keywords', 'Reviews']
            df1 = df_dom_topic.head(5)

            pd.options.display.max_colwidth = 120

            sorted_topics_df_mallet = pd.DataFrame()
            sorted_topics_out_df_grpd = df_topic_sents_keywords.groupby('Dom_Topic')

            for i, grp in sorted_topics_out_df_grpd:
                sorted_topics_df_mallet = pd.concat([sorted_topics_df_mallet, 
                                                        grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                                        axis=0)

            # Reset Index    
            sorted_topics_df_mallet.reset_index(drop=True, inplace=True)

            # Format
            sorted_topics_df_mallet.columns = ['Topic', "Contribution", "Keywords", "Most Representative Entry"]

            # Show
            df = sorted_topics_df_mallet.head(5)

            #main_themes = sorted_topics_df_mallet.head(10)  
            return render_template('insight.html', tables=[df.to_html(classes='data')], titles=df.columns.values)  
    
        # if request.form.get('submit2') == 'Topic Distribution':
        #     mysql.connection.cursor()
        #     query = ("SELECT * FROM journals")
        #     df = pd.read_sql(query, mysql.connection) 
        #     df['processed_text'] = df.apply(lambda row: nltk.word_tokenize(row['processed_text']), axis=1)
        #     clean_tokens = df['processed_text']
        #     id2word = corpora.Dictionary(clean_tokens)
        #     corpus = [id2word.doc2bow(text) for text in clean_tokens]
        #     lda_model = gensim.models.LdaModel(corpus ,id2word=id2word, num_topics=6, alpha='auto', eval_every=5, random_state=300,update_every=1,chunksize=10,passes=10,iterations=300,per_word_topics=True)
        #     p = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        #     pyLDAvis.save_html(p, 'LDA_Visualization.html')
        #     return render_template('insight.html')

        if request.form.get('submit2') == 'Topic Distribution':
            return render_template('LDA_Visualisation.html')
        
        if request.form.get('action1') == 'Full List':
            #cursor for mysql creates post request to enter db
            cur = mysql.connection.cursor()
            #display the output into the display list
            resultValue = cur.execute("SELECT * FROM journals")
            if resultValue > 0:
                journals = cur.fetchall()
            return render_template('insight.html',journals=journals)

        if request.form.get('action2') == 'Positive Entries':
            #cursor for mysql creates post request to enter db
            cur = mysql.connection.cursor()
            #display the output into the display list
            resultValue = cur.execute("SELECT * FROM journals WHERE sentiment='positive'")
            if resultValue > 0:
                journals = cur.fetchall()
            return render_template('insight.html',journals=journals)

        if request.form.get('action3') == 'Negative Entries':
            #cursor for mysql creates post request to enter db
            cur = mysql.connection.cursor()
            #display the output into the display list
            resultValue = cur.execute("SELECT * FROM journals WHERE sentiment='negative'")
            if resultValue > 0:
                journals = cur.fetchall()
            return render_template('insight.html',journals=journals)

        if request.form.get('submit_button1') == 'All Entries':
            mysql.connection.cursor()
            query = ("SELECT * FROM journals")
            df = pd.read_sql(query, mysql.connection)
            journal_entry = df['journal_entry']
            long_string = (''.join(str(journal_entry.tolist())))
            # Create WordCloud object
            wordcloud = WordCloud(background_color="white", max_words=4500, contour_width=6, contour_color='blue', width=800, height=400)
            # Generate the cloud
            wordcloud.generate(long_string)
            wordcloud.to_file("wordcloud.png")
            filename = Image.open("wordcloud.png")
            main_themes = filename.show()
            return render_template('insight.html', value=main_themes)

        if request.form.get('submit_button2') == 'Positive':
            mysql.connection.cursor()
            query = ("SELECT journal_entry FROM journals WHERE sentiment = 'positive'")
            df = pd.read_sql(query, mysql.connection)
            journal_entry = df['journal_entry']
            long_string = (''.join(str(journal_entry.tolist())))
            # Create WordCloud object
            wordcloud = WordCloud(background_color="white", max_words=700, contour_width=6, contour_color='pink', width=800, height=400)
            # Generate the cloud
            wordcloud.generate(long_string)
            wordcloud.to_file("wordcloud.png")
            filename = Image.open("wordcloud.png")
            main_themes = filename.show()
            return render_template('insight.html', value=main_themes)
        
        if request.form.get('submit_button3') == 'Neutral':
            mysql.connection.cursor()
            query = ("SELECT journal_entry FROM journals WHERE sentiment = 'neutral'")
            df = pd.read_sql(query, mysql.connection)
            journal_entry = df['journal_entry']
            long_string = (''.join(str(journal_entry.tolist())))
            # Create WordCloud object
            wordcloud = WordCloud(background_color="white", max_words=500, contour_width=6, contour_color='blue', width=800, height=400)
            # Generate the cloud
            wordcloud.generate(long_string)
            wordcloud.to_file("wordcloud.png")
            filename = Image.open("wordcloud.png")
            main_themes = filename.show()
            return render_template('insight.html', value=main_themes)

        if request.form.get('submit_button4') == 'Negative':
            query = ("SELECT journal_entry FROM journals WHERE sentiment = 'negative'")
            df = pd.read_sql(query, mysql.connection)
            journal_entry = df['journal_entry']
            long_string = (''.join(str(journal_entry.tolist())))
            # Create WordCloud object
            wordcloud = WordCloud(background_color="white", max_words=500, contour_width=6, contour_color='red', width=800, height=400)
            # Generate the cloud
            wordcloud.generate(long_string)
            wordcloud.to_file("wordcloud.png")
            filename = Image.open("wordcloud.png")
            main_themes = filename.show()
            return render_template('insight.html', value=main_themes)
         
    else:   
        return render_template('insight.html')

@app.route('/journal', methods=['GET', 'POST'])
def journal():
    #Post means info should be stored in database
    #Get means info should be displayed directly to user
    if request.method == 'POST':
        #Fetch the form data
        #Create variable to store form data
        journals = request.form.to_dict()
    
        entry_date = journals['entry_date']
        journal_entry = journals['journal_entry']

        sdf = vec.transform([journal_entry]).reshape(1, -1)
        sentiment = sentiment_model.predict(sdf)

        journals['sentiment'] = sentiment

        text = journal_entry

        def preprocess(text):
            
            #convert text to lowercase
            text = text.lower()
            
            #remove urls. done using regular expression
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
            
            #remove punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))
            
            #remove any users referenced and remove hashtags, using regular expression
            text = re.sub(r"\@\'w+|\#", "", text)
            text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
            
            #remove stopwords
            review_tokens = word_tokenize(text)
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
 
        processed_text = preprocess(text)

        #cursor for mysql creates post request to enter db
        cur = mysql.connection.cursor()
        #insert the values with sentiment attribute back into database
        cur.execute("INSERT INTO journals(entry_date, journal_entry, sentiment, processed_text) VALUES(%s, %s, %s, %s)",(entry_date, journal_entry, sentiment, processed_text))
        #commit changes to db
        mysql.connection.commit()       
    return render_template('journal.html')

@app.route('/insight', methods=['GET', 'POST'])
def display():
    #Post means info should be stored in database
    #Get means info should be displayed directly to user
    if request.method == 'POST':
        if request.form.get('action1') == 'See List':
            #cursor for mysql creates post request to enter db
            query = ("SELECT entry_date, journal_entry, sentiment FROM journals")
            df = pd.read_sql(query, mysql.connection)
            df = pd.DataFrame
            journals = df.style
            return render_template('insight.html',journals=journals)

        if request.form.get('submit_button1') == 'All Entries':
            mysql.connection.cursor()
            query = ("SELECT * FROM journals")
            df = pd.read_sql(query, mysql.connection)
            journal_entry = df['journal_entry']
            long_string = (''.join(str(journal_entry.tolist())))
            # Create WordCloud object
            wordcloud = WordCloud(background_color="white", max_words=4500, contour_width=6, contour_color='blue', width=800, height=400)
            # Generate the cloud
            wordcloud.generate(long_string)
            wordcloud.to_file("wordcloud.png")
            filename = Image.open("wordcloud.png")
            main_themes = filename.show()
            return render_template('insight.html', value=main_themes)

        if request.form.get('submit_button2') == 'Positive':
            mysql.connection.cursor()
            query = ("SELECT journal_entry FROM journals WHERE sentiment = 'positive'")
            df = pd.read_sql(query, mysql.connection)
            journal_entry = df['journal_entry']
            long_string = (''.join(str(journal_entry.tolist())))
            # Create WordCloud object
            wordcloud = WordCloud(background_color="white", max_words=700, contour_width=6, contour_color='pink', width=800, height=400)
            # Generate the cloud
            wordcloud.generate(long_string)
            wordcloud.to_file("wordcloud.png")
            filename = Image.open("wordcloud.png")
            main_themes = filename.show()
            return render_template('insight.html', value=main_themes)
        
        if request.form.get('submit_button3') == 'Neutral':
            mysql.connection.cursor()
            query = ("SELECT journal_entry FROM journals WHERE sentiment = 'neutral'")
            df = pd.read_sql(query, mysql.connection)
            journal_entry = df['journal_entry']
            long_string = (''.join(str(journal_entry.tolist())))
            # Create WordCloud object
            wordcloud = WordCloud(background_color="white", max_words=500, contour_width=6, contour_color='blue', width=800, height=400)
            # Generate the cloud
            wordcloud.generate(long_string)
            wordcloud.to_file("wordcloud.png")
            filename = Image.open("wordcloud.png")
            main_themes = filename.show()
            return render_template('insight.html', value=main_themes)

        if request.form.get('submit_button4') == 'Negative':
            query = ("SELECT journal_entry FROM journals WHERE sentiment = 'negative'")
            df = pd.read_sql(query, mysql.connection)
            journal_entry = df['journal_entry']
            long_string = (''.join(str(journal_entry.tolist())))
            # Create WordCloud object
            wordcloud = WordCloud(background_color="white", max_words=500, contour_width=6, contour_color='red', width=800, height=400)
            # Generate the cloud
            wordcloud.generate(long_string)
            wordcloud.to_file("wordcloud.png")
            filename = Image.open("wordcloud.png")
            main_themes = filename.show()
            return render_template('insight.html', value=main_themes)
         
    else:   
        return render_template('insight.html')

if __name__ == '__main__':
    #this line enables the code to refresh
    app.run(debug=True)
