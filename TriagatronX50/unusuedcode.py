
# @app.route('/insight', methods=['GET', 'POST'])
# def insight():
#     if request.method == 'GET':
            
#             lda = gensim.models.ldamodel.LdaModel.load(temp_file)

#             mysql.connection.cursor()
#             query = ("SELECT * FROM journals")
#             df = pd.read_sql(query, mysql.connection)
#             journals_df = pd.DataFrame(df)
#             journal_entry = journals_df['journal_entry']
#             entryarray = journal_entry.to_numpy()
#             # entrylist = journal_entry.values.tolist()
#             # my_array = entrylist.split(',')
#             other_corpus = [common_dictionary.doc2bow(text) for text in entryarray]
#             unseen_doc = other_corpus[0]
#             vector = lda[unseen_doc]
#             lda.update(other_corpus)
#             vector = lda[unseen_doc]
            

#             main_themes = vector
        
#     return render_template('insight.html', value=main_themes)


# @app.route('/insight', methods=['GET', 'POST'])
# def insight():
#     if request.method == 'GET':
#             mysql.connection.cursor()
#             query = ("SELECT * FROM journals")
#             df = pd.read_sql(query, mysql.connection)
#             journals_df = pd.DataFrame(df)
#             journal_entry = journals_df['journal_entry']
#             journals_dict = journals_df.to_dict()
            
#             # def sent_to_words(sentences):
#             #     for sent in sentences:
#             #         sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
#             #         sent = re.sub('\s+', ' ', sent)  # remove newline chars
#             #         sent = re.sub("\'", "", sent)  # remove single quotes
#             #         sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
#             #         yield(sent)
#             # Convert to list
#             corpus = journal_entry.values.tolist()
#             #processed = list(sent_to_words(data))                   
#             #corpus = [journals_dict.doc2bow(text) for text in data]
#             lda = gensim.models.ldamodel.LdaModel.load(temp_file)
#             vis = gensimvis.prepare(lda, corpus=corpus, dictionary=journals_dict)
#             main_themes = pyLDAvis.save_html(vis, 'LDA_Visualization.html')
        
#     return render_template('insight.html', value=main_themes)


#import spacy
#nlp = spacy.load('en', disable=['parser', 'ner'])

# @app.route('/insight', methods=['GET', 'POST'])
# def insight():
#     if request.method == 'GET':
#         # if request.form['submit_button'] == 'Main Themes':
#             lda_model = gensim.models.ldamodel.LdaModel.load(temp_file)
#             mysql.connection.cursor()
#             query = ("SELECT journal_entry FROM journals")
#             df = pd.read_sql(query, mysql.connection)
#             df1 = df.apply(lambda x: (x)[0].toarray())
#             #df1 =  bytes(df1,'UTF-8')
#             #df1 = entry.split('')
#             corpus = [common_dictionary.doc2bow(text) for text in df]
#             #corpus = new_corpus #lda_model = gensim.models.ldamodel.LdaModel(new_corpus, num_topics=3, alpha='auto', eval_every=5) 
#             #vis = gensimvis.prepare(lda, new_corpus, common_dictionary)
#             main_themes = corpus
#             #lda_model.print_topics()
#             #pyLDAvis.save_html(vis, 'LDA_Visualization.html')
#             # print the topics identified in the html page
#             #main_themes = show(LDA_Visualization)
#         # return render_template('insight.html', value=main_themes)        
#     return render_template('insight.html', value=main_themes) 