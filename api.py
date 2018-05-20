# FORCE CPU USAGE - this VPS does not have a GPU
# Note: When I loaded this using AVX2 instructions I got a strange error.
# probably because it's a virtual CPU
# in any case, if you have the same problem, install this TF whl 
# https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.6.0/py36/CPU/sse2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# generic
from flask import Flask, abort, request
import flask

import datetime
import time
import numpy as np
import emoji
import re
import logging
import sys
import os
from pathlib import Path
# twitter related
import tweepy
import validators
from tld import get_tld
from oauth import consumer_key, consumer_secret, access_token, access_token_secret
# keras / ai related
from constants import MAX_VECTOR_COUNT
from sklearn.externals import joblib
import spacy
import tensorflow as tf
import keras
from keras.models import load_model

# set up logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
# Init- must occur in global scope, since other funcs will need them later
emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
r = re.compile('|'.join(re.escape(p) for p in emojis_list))
# approved senders file io logic
from senders import *


def reply_percentages(status, model, scaler):
    username = status._json['user']['screen_name']
    status_id = status._json['id_str']
    retweeted = status._json['is_quote_status']
    print(retweeted)
    notified_senders = get_notified_senders()
    if not retweeted:
        if username in notified_senders:
            logging.warning("Not sending notification for {0} because they are already in the notified_senders list.".format(username))
            return
        api.update_status(status='@{0} It looks like you tweeted text at me without quoting another status. Quote someone else to see the results properly. \nLearn more at http://trollspotter.com'.format(username), in_reply_to_status_id=status_id)
        logging.debug("Notifying user {0} that they need to retweet. This will be the last notification they get.".format(username))
        add_notified_sender(username)
    else:
        quoted_status = status._json['quoted_status']['text']
        tokenized_raveled_padded = tokenize_ravel_pad(quoted_status)
        scaled = scale_tweet(tokenized_raveled_padded)
        predicted = predict_scaled_tweet_threaded(scaled)
        predicted_list = predicted[0].tolist()
        logging.info(predicted_list)
        logging.info(predicted_list[0])
        logging.info(predicted_list[1])
        api.update_status(status='@{0} Regular: {1}\nTroll: {2}\nLearn more at http://trollspotter.com'.format(username, predicted_list[0], predicted_list[1]), in_reply_to_status_id=status_id)

# create a class inheriting from the tweepy  StreamListener
# tried breaking this up into another file, it's here because of some strange scoping / multithreading issues otherwise
class BotStreamer(tweepy.StreamListener):
    def on_status(self, status):
        logging.info("Processing a new status " + status._json['text'])
        logging.info("Processing a new status " + status._json['text'])
        # entities provide structured data from Tweets including resolved URLs, media, hashtags and mentions without having to parse the text to extract that information
        reply_percentages(status, model, scaler)
        
    def on_error(self, error):
        logging.error("ERROR! " + str(error))
        if status_code == 420:
            logging.error("We are rate limited!")
            return True
    def on_disconnect(self, error):
        logging.warning("Disconnected from service!")

application = app = Flask(__name__)
now = datetime.datetime.now()
logging.info("Preprocessing begins at " + str(now))
logging.info("Loading SpaCy's large vector set. " + str(now))
nlp = spacy.load('en_vectors_web_lg')
now = datetime.datetime.now()
logging.info("Finished loading. " + str(now))
logging.info("Loading scaler from trained data.")
scaler = joblib.load("scaler.pkl")
logging.info("Loading complete.")

# twitter api setup
logging.info("Establishing Twitter API connection.")
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

logging.info("Loading Keras model...")
model = load_model("troll_model.h5")
# needed for multithreading support
model._make_predict_function()
graph = tf.get_default_graph()
logging.info("Load complete. Loading stream server.")
botStreamListener = BotStreamer()
stream = tweepy.Stream(auth, botStreamListener)
stream.filter(track=['@Troll_Spotter'], async=True, stall_warnings=True)
logging.info("Async thread creation complete!")

logging.info("Listening for requests...")
def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')


def pad_tweet(tweet):
    if len(tweet) > MAX_VECTOR_COUNT:
        print("Trimming tweet vectors with vector length " + str(len(tweet)))
        tweet = tweet[:MAX_VECTOR_COUNT]
        # trim, but it's highly unlikely there are that many tokens
    else:
        # we need to add fake empty tokens to pad
        needed_empty = MAX_VECTOR_COUNT - len(tweet)
        tweet = padarray(tweet, MAX_VECTOR_COUNT)
    return tweet

    
def tokenizeTweet(tweet):
    output = []
    rowTokens = []
    tokens = nlp(tweet)
    for tok in tokens:
        output.append(tok.vector)
    return output

    
def tokenize_ravel_pad(tweet):
    tokens = tokenizeTweet(filterEmojis(tweet))
    raveled = np.concatenate(tokens).ravel()
    padded = pad_tweet(raveled)
    return padded
    
    
def scale_tweet(tweet):
    return scaler.transform(tweet.reshape(1, -1))
    
    
def filterEmojis(s):
    s = re.sub(r, ' ', s)
    return s

    
def predict_scaled_tweet(tweet):
    predicted = model.predict(tweet)
    logging.info("Predicted {0}".format(predicted))
    return predicted
def predict_scaled_tweet_threaded(tweet):
    with graph.as_default():
        return model.predict(tweet)
    
@app.route('/', methods=['GET'])
def heartbeat():
    return "<html><body>Alive and well!</body></html>"
    
    
@app.route('/predict', methods=['GET'])
def vector():
    target = request.args.get('target')
    if not target:
        logging.error("An invalid request was submitted to the API.")
        abort(400)
    
    logging.info("Processing new request " + target)
    final_target = "" #the text the AI will actually see; retrieved from APIs if url
    if validators.url(target):
        if "twitter" in get_tld(target):
            logging.info("Flagged " + target + " as a tweet.")
            final_target = get_tweet_raw_text(api, target)
            
            #if api.rate_limit_status()['resources']['statuses']['statuses/lookup'] >= 0:
            #else:
            #    logging.error("We are rate limited!")
            #    abort(420)
            # removed this because rate_limit_status is a gigantic json object
        else:
            logging.error("A url was submitted " + target + " but it is not a TLD matching the API filter.")
            abort(406) #406 not accepted
    else:
        final_target = target
    tokenize_ravel_padded = tokenize_ravel_pad(final_target)
    scaled = scale_tweet(tokenize_ravel_padded)
    predicted = predict_scaled_tweet(scaled)
    return flask.jsonify(predicted.tolist())

    
def get_tweet_raw_text(api, url):
    idVal = url.split('/')[-1]
    public_tweet = api.get_status(idVal, tweet_mode='extended')
    logging.info("Retrieved tweet: " + public_tweet.full_text + " from url " + url)
    return public_tweet.full_text

    
    
    
def main():
    #app.run(debug=False, use_reloader=False)
    app.run(debug=False, use_reloader=False, host="127.0.0.1")
if __name__ == '__main__':
    main()

    