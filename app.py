from flask import Flask, render_template, request, redirect, url_for, flash
import nltk
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle
import googleapiclient.discovery
import googleapiclient.errors
from google_play_scraper import app,Sort, reviews_all
import requests
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('stopwords')
data = pd.read_csv('IMDB-Dataset.csv')
data.sentiment.replace('positive',1,inplace=True)
data.sentiment.replace('negative',0,inplace=True)
print('1')
# user define function
# Scrape the data
def getdata(url):
	r = requests.get(url)
	return r.text
def html_code(url):

	# pass the url
	# into getdata function
	htmldata = getdata(url)
	soup = BeautifulSoup(htmldata, 'html.parser')

	# display html code
	return soup
def cus_rev(soup):
	# find the Html tag
	# with find()
	# and convert into string
	data_str = ""

	for item in soup.find_all("span", class_="a-size-base review-text review-text-content"):
		data_str = data_str + item.get_text()

	result = data_str.split("\n")
	return result

def amazonextract(url):
    soup = html_code(url)

    rev_data = cus_rev(soup)
    rev_result = []
    for i in rev_data:
        if i is "":
            pass
        else:
            rev_result.append(i)

    return rev_result

def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem
def to_lower(text):
    return text.lower()
def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]
def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])
data.review = data.review.apply(clean)
data.review = data.review.apply(is_special)
data.review = data.review.apply(to_lower)
data.review = data.review.apply(rem_stopwords)
data.review = data.review.apply(stem_txt)

X = np.array(data.iloc[:,0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(data.review).toarray()

trainx,testx,trainy,testy = train_test_split(X,y,test_size=0.2,random_state=9)

gnb,mnb,bnb = GaussianNB(),MultinomialNB(alpha=1.0,fit_prior=True),BernoulliNB(alpha=1.0,fit_prior=True)
gnb.fit(trainx,trainy)
mnb.fit(trainx,trainy)
bnb.fit(trainx,trainy)

#Output
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

ypg = gnb.predict(testx)
ypm = mnb.predict(testx)
ypb = bnb.predict(testx)

pickle.dump(bnb,open('model1.pkl','wb'))
def ytextract(input):
    api_service_name = "youtube"
    api_version = "v3"
    # this is api key for youtube
    DEVELOPER_KEY = "AIzaSyArftdDLecWXRFm3hWkjBC79Rr_bTK8qCA"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=input,
        maxResults=100
    )
    response = request.execute()

    comments = []
    for item in response['items']:
        comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment_text)


    print(comments[2])
    return comments

print('2')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('login.html')

@app.route('/scrape')
def scrape():
    return render_template('scrape.html')

@app.route('/playinput')
def playinput():
    return render_template('playinput.html')

@app.route('/ytinput')
def ytinput():
    return render_template('ytinput.html')

@app.route('/amazonip')
def amazonip():
    return render_template('amazonip.html')

@app.route('/processplay',methods=['POST'])
def processplay():
    inputdata = request.form['input_data']
    print(inputdata)
    # name = 'com.fantome.penguinisle'
    result = reviews_all(
        app_id=inputdata,
        sleep_milliseconds=0,  # defaults to 0
        lang='en',  # defaults to 'en'
        country='us',  # defaults to 'us'
        sort=Sort.MOST_RELEVANT,  # defaults to Sort.MOST_RELEVANT
        filter_score_with=5  # defaults to None(means all score)
    )

    li = []
    for i in result:
        content = i
        li.append(content['content'])
    # len(li)
    reviews=li

    pc = []
    nc = []
    ## put lk when run actual program
    for i in reviews:
        f1 = clean(i)
        f2 = is_special(f1)
        f3 = to_lower(f2)
        f4 = rem_stopwords(f3)
        f5 = stem_txt(f4)

        bow, words = [], word_tokenize(f5)
        for word in words:
            bow.append(words.count(word))
        # np.array(bow).reshape(1,3000)
        # bow.shape
        word_dict = cv.vocabulary_
        pickle.dump(word_dict, open('bow.pkl', 'wb'))
        inp = []
        for i in word_dict:
            inp.append(f5.count(i[0]))
        y_pred = bnb.predict(np.array(inp).reshape(1, 1000))
        result = y_pred[0]

        if result == 1:
            pc.append(i)
        else:
            nc.append(i)
    tpc = len(pc)
    tnc = len(nc)
    print('total number of positive reviews', tpc)
    print('total number of negative reviews', tnc)

    tpc_percentage = (tpc / (tpc + tnc)) * 100  # Calculate percentage of positive reviews
    tnc_percentage = (tnc / (tpc + tnc)) * 100  # Calculate percentage of negative reviews

    if tpc > tnc:
        ipdata='This is nice app, you can use'
        return render_template('playop.html', my_string=ipdata, tpc=tpc, tnc=tnc, tpc_percentage=tpc_percentage, tnc_percentage=tnc_percentage)
    else:
        ipdata="This is not nice app, your wish to use"
        return render_template('playop.html', my_string=ipdata, tpc=tpc, tnc=tnc, tpc_percentage=tpc_percentage, tnc_percentage=tnc_percentage)


@app.route('/processyt',methods=['POST'])
def processyt():
    input = request.form['input_data']
    print(input)

    comments=ytextract(input)

    reviews = comments
    pc = []
    nc = []
    ## put lk when run actual program
    for i in reviews:
        f1 = clean(i)
        f2 = is_special(f1)
        f3 = to_lower(f2)
        f4 = rem_stopwords(f3)
        f5 = stem_txt(f4)

        bow, words = [], word_tokenize(f5)
        for word in words:
            bow.append(words.count(word))
        # np.array(bow).reshape(1,3000)
        # bow.shape
        word_dict = cv.vocabulary_
        pickle.dump(word_dict, open('bow.pkl', 'wb'))
        inp = []
        for i in word_dict:
            inp.append(f5.count(i[0]))
        y_pred = bnb.predict(np.array(inp).reshape(1, 1000))
        result = y_pred[0]

        if result == 1:
            pc.append(i)
        else:
            nc.append(i)
    tpc = len(pc)
    tnc = len(nc)
    print('total number of positive reviews', tpc)
    print('total number of negative reviews', tnc)

    tpc_percentage = (tpc / (tpc + tnc)) * 100  # Calculate percentage of positive reviews
    tnc_percentage = (tnc / (tpc + tnc)) * 100  # Calculate percentage of negative reviews

    if tpc > tnc:
        ipdata='Positive comments, you can watch'
        return render_template('ytop.html', my_string=ipdata, tpc=tpc, tnc=tnc, tpc_percentage=tpc_percentage, tnc_percentage=tnc_percentage)
    else:
        ipdata="Negative comments, watch your wish"
        return render_template('ytop.html', my_string=ipdata, tpc=tpc, tnc=tnc, tpc_percentage=tpc_percentage, tnc_percentage=tnc_percentage)

@app.route('/processamazon',methods=['POST'])
def processamazon():
    input = request.form['input_data']
    print(input)

    comments=amazonextract(input)

    reviews = comments
    pc = []
    nc = []
    ## put lk when run actual program
    for i in reviews:
        f1 = clean(i)
        f2 = is_special(f1)
        f3 = to_lower(f2)
        f4 = rem_stopwords(f3)
        f5 = stem_txt(f4)

        bow, words = [], word_tokenize(f5)
        for word in words:
            bow.append(words.count(word))
        # np.array(bow).reshape(1,3000)
        # bow.shape
        word_dict = cv.vocabulary_
        pickle.dump(word_dict, open('bow.pkl', 'wb'))
        inp = []
        for i in word_dict:
            inp.append(f5.count(i[0]))
        y_pred = bnb.predict(np.array(inp).reshape(1, 1000))
        result = y_pred[0]

        if result == 1:
            pc.append(i)
        else:
            nc.append(i)
    tpc = len(pc)
    tnc = len(nc)
    print('total number of positive reviews', tpc)
    print('total number of negative reviews', tnc)

    tpc_percentage = (tpc / (tpc + tnc)) * 100  # Calculate percentage of positive reviews
    tnc_percentage = (tnc / (tpc + tnc)) * 100  # Calculate percentage of negative reviews

    if tpc > tnc:
        ipdata='This is nice product, you can buy'
        return render_template('amazonop.html', my_string=ipdata, tpc=tpc, tnc=tnc, tpc_percentage=tpc_percentage, tnc_percentage=tnc_percentage)
    else:
        ipdata="This is not nice product,your wish to buy"
        return render_template('amazonop.html', my_string=ipdata, tpc=tpc, tnc=tnc, tpc_percentage=tpc_percentage, tnc_percentage=tnc_percentage)

if __name__ == '__main__':
    app.run(debug=True)