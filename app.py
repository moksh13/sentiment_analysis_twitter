from flask import Flask, render_template, flash, request, url_for
import numpy as np 
import pandas as pd 
import re
import os
import tensorflow as tf 
from numpy import array
from keras.datasets import imdb

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model



app = Flask(__name__, template_folder='template')

def init():
	global model ,graph
	#### load the pre-trained keras model####
	model = load_model('model.h5', compile=False)
	graph = tf.get_default_graph()


###### code for sentiment analysis #######

@app.route('/',methods=['GET','POST'])
def home():

	return render_template("home.html")


@app.route('/sentiment_analysis_pre', methods =['POST',"GET"])
def sent_pre():
	if request.method=='POST':
		text = request.form['text']
		Sentiment = ''
		max_len = 300
		word_to_id = imdb.get_word_index()
		strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
		text = text.lower().replace("<br />"," ")
		text = re.sub(strip_special_chars, "", text.lower())
        

		words = text.split()  # split string into a list
		x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
		x_test = sequence.pad_sequences(x_test, maxlen=300)
		
    
		vector = np.array([x_test.flatten()])
		
		with graph.as_default():

			probability = model.predict(array([vector][0]))[0][0]
			class1 = model.predict_classes(array([vector][0]))[0][0]
		if class1 == 0:
			sentiment = 'Negative'
		else:
			sentiment = 'Positive'
	return render_template('home.html', text=text, sentiment=sentiment, probability=probability)

### code for sentiment analysis ###


if __name__ == "__main__":
	init()
	app.run(debug=True, threaded=False)
