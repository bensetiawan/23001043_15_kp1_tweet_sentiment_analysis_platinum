import pandas as pd
import re
import pickle
import sqlite3
import nltk

from flask import Flask, request, jsonify
from flasgger import Swagger,LazyJSONEncoder
from flasgger import swag_from

from nltk.corpus import stopwords as stopword
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from flask import request
app = Flask(__name__)

#db_connection
def db_connection():
    conn = None
    try:
        conn = sqlite3.connect('db_alay.sqlite')
    except sqlite3.error as e:
        print(e)
    return conn

app.json_encoder = LazyJSONEncoder

#swagger template
swagger_template = dict(
info = {
    'title': 'Project : Text and Tweets Sentiment Analysis',
    'version': '1.0.0',
    'description': 'API Documentation for Text and Tweets Sentiment Analysis'
    }
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template,config=swagger_config)

connection1 = db_connection()
#run_query1 = connection.cursor()

# read file kamusalay
kamusalay_df = pd.read_csv('Dataset/new_kamusalay.csv', names = ['ALAY', 'TIDAK_ALAY'], encoding = 'latin1')
# store to database
kamusalay_df.to_sql("replace_alay",  connection1, if_exists='append', index=False)
# Get data kamusalay from database
temp_kamusalay_df = pd.read_sql_query("SELECT ALAY,TIDAK_ALAY FROM replace_alay", connection1)

#Change Dataframe to Dictionary
dict_alay = {
    'ALAY':[],
    'TIDAK_ALAY':[]
}
for i in temp_kamusalay_df.itertuples():
  dict_alay['ALAY'].append(i.ALAY)
  dict_alay['TIDAK_ALAY'].append(i.TIDAK_ALAY)


# 1. remove USER,RT,URL
def remove_user_rt_url (str):
    string = re.sub(r'USER|\bRT\b|URL',' ',str)
    return string

#2 buat lower case
def lower_case (str):
    string = str.lower()
    return string

#3 remove /n
def remove_n (str):
    string =  re.sub(r'\\n',' ',str)
    return string

#4 remove emoji
def remove_emo2 (str):
    pattern = re.compile(r'[\\x]+[a-z0-9]{2}')
    string = re.sub(pattern,'',str)
    return string

#5 Remove Link
# remove link (http|https)
def remove_link (str):
    pattern = re.compile(r'www\S+|http\S+')
    string =  re.sub(pattern,' ',str)
    return string

#6 Hapus sisa karakter
# hapus special character dan pertahankan text
def remove_character(str):
    string = re.sub(r'[^a-zA-Z]+',' ',str)
    return string

#7 repalce alay
def replace_alay(str):
    for i in range(0,len(temp_kamusalay_df)-1): 
        alay = dict_alay['ALAY'][i]
        if (' ' + alay + ' ') in (' ' + str + ' '):
            replace = dict_alay['TIDAK_ALAY'][i]
            str = re.sub(r'\b{}\b'.format(alay),replace,str)
    return str

#8 remove extra space
def remove_extra_space (str):
    str = re.sub('  +', ' ', str)
    str = str.strip()
    return str

def text_cleansing(str):
    str = remove_user_rt_url (str)
    str = lower_case (str)
    str = remove_n (str)
    str = remove_emo2 (str)
    str = remove_link (str)
    str = remove_character(str)
    str = remove_extra_space (str)
    str = replace_alay(str)
    str = remove_extra_space (str)
    return str


## Preprocessing Text

factory = StemmerFactory()
stemer = factory.create_stemmer()

list_stopwords_id = stopword.words('indonesian')
list_stopwords_en = stopword.words('english')

list_stopwords_id.extend(list_stopwords_en)
list_stopwords_id.extend(['ya','yg','ga','yuk','dah','nya','duh','sih'])

## Remove spesific stopword from list
not_stopwords = {'enggak', 'tidak'}
list_stopwords_id = set([word for word in list_stopwords_id if word not in not_stopwords])

#1 Tokenisasi
def tokenize (text):
  token = word_tokenize(text)
  return token

#2 StopWord removal
def stop_words(text):
  after_stopwords = [word for word in text if not word in list_stopwords_id]
  return after_stopwords

#3 Stemming
def stemming (text):
  after_stemming = [stemer.stem(word) for word in text]
  return after_stemming

#4 Token to sentence
def sentence(list_words):
  sentence = ' '.join(word for word in list_words)
  return sentence


def text_preprocesing(text):
  text = tokenize(text)
  text = stop_words(text)
  text = stemming(text)
  text = sentence(text)
  return text

file_train = open('text_preprocessing.pickle','rb')
data_train = pickle.load(file_train)
file_train.close()

tfidf = TfidfVectorizer()
tfidf.fit(data_train.tolist())

tokenisasi = Tokenizer(num_words = 100000, oov_token = 'OOV')
tokenisasi.fit_on_texts(data_train)

def texts_sequences (text):
  sequence = tokenisasi.texts_to_sequences(text)
  return sequence

def pading(sequences):
  pad = pad_sequences(sequences, maxlen = 55)
  return pad

def sequences_padding(text):
  sequences = texts_sequences(text)
  print(sequences)
  pad = pading(sequences)
  return pad

model_LSTM = load_model('Model_LSTM/model_LSTM.h5')

file_mlp = open('Model_MLP/model_MLP.pickle','rb')
model_MLP = pickle.load(file_mlp)
file_mlp.close()

def predict_sentiment_LSTM(text):
  text = [text_preprocesing(text)]
  pad = sequences_padding(text)
  predict = model_LSTM.predict(pad)
 # print(predict)
  for i in predict:
    if i[0] > i[1] and i[0] > i[2]:
      output = 'Negative'
    elif i[1] > i[0] and i[1] > i[2]:
      output = 'Neutral'
    elif i[2] > i[0] and i[2] > i[1]:
      output = 'Positive'
  return output

def predict_sentiment_MLP(text):
  preproces = [text_preprocesing(text)]
  text_transform = tfidf.transform(preproces)
  predict = model_MLP.predict(text_transform)
  for i in predict:
    if i[0] == 1:
      output = 'Negative'
    elif i[1] == 1:
      output = 'Neutral'
    elif i[2] == 1:
      output = 'Positive'
    ## Condition : [[0 0 0]]
    else:
      output = 'Neutral'
  return output

@app.route('/', methods=['GET'])
@swag_from("docs/team.yml", methods=['GET'])
def hello_world():
    json_response = {
        'Anggota_1': "Bianda Shafira",
        'Anggota_2' : "Syarifudien Zuhdi",
        'Anggota_3' : "Vieri Valerian",
        'Anggota_4' : "Ruben Setiawan",
        'Project': "Tweet Sentiment Analysis",
        'Github': "https://github.com/bensetiawan",
        'Challange': "PLATINUM - CHALLANGE"
    }
    response_data = jsonify(json_response)
    return response_data

@app.route('/text_sentiment_LSTM', methods=['POST'])
@swag_from('docs/text_LSTM.yml', methods=['POST'])
def text_sentiment_LSTM():
    text = request.form.get('text')
    temp =  text
    cleansing_text = text_cleansing(temp)
    sentiment = predict_sentiment_LSTM(cleansing_text)

    json_response = {
        'Description':'Text Sentiment Analysis',
        'Input_Text':temp,
        'Sentiment' : sentiment
    }

    response_data  =  jsonify(json_response)
    return response_data

@app.route('/Tweet_Sentiment_LSTM', methods = ['POST'])
@swag_from('docs/file_LSTM.yml', methods=['POST'])
def  tweet_file_cleansing():
    file = request.files.getlist('file')[0]

    tweet_df = pd.read_csv(file, encoding = 'windows-1250',usecols = [0])

    tweet_df.drop_duplicates()

    tweet_df['Text_Cleansing'] = tweet_df['Tweet'].apply(text_cleansing)

    Tweet_Sentiment = [
        dict(Tweet = tweet_df['Text_Cleansing'].loc[i], Tweet_Sentiment = predict_sentiment_LSTM(tweet_df['Text_Cleansing'].loc[i]))
        for i in tweet_df['Tweet'].index.values
    ]
    
    if Tweet_Sentiment is not None:
        return jsonify(Tweet_Sentiment)

@app.route('/text_sentiment_MLP', methods=['POST'])
@swag_from('docs/text_MLP.yml', methods=['POST'])
def text_sentiment_MLP():
    text = request.form.get('text')
    temp =  text
    cleansing_text = text_cleansing(temp)
    sentiment = predict_sentiment_MLP(cleansing_text)

    json_response = {
        'Description':'Text Sentiment Analysis',
        'Input_Text':temp,
        'Sentiment' : sentiment
    }

    response_data  =  jsonify(json_response)
    return response_data


@app.route('/Tweet_Sentiment_MLP', methods = ['POST'])
@swag_from('docs/file_MLP.yml', methods=['POST'])
def  file_sentiment_MLP():
    file = request.files.getlist('file')[0]

    tweet_df = pd.read_csv(file, encoding = 'windows-1250', usecols = [0])

    tweet_df.drop_duplicates()

    tweet_df['Text_Cleansing'] = tweet_df['Tweet'].apply(text_cleansing)

    Tweet_Sentiment = [
        dict(Tweet = tweet_df['Text_Cleansing'].loc[i], Tweet_Sentiment = predict_sentiment_MLP(tweet_df['Text_Cleansing'].loc[i]))
        for i in tweet_df['Tweet'].index.values
    ]
    
    if Tweet_Sentiment is not None:
        return jsonify(Tweet_Sentiment)

if __name__ == "__main__":
    app.run(debug=True)

   