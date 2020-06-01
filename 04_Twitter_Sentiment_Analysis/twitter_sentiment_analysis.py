from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from sentiment_analysis import *

# Keys and tokens for the Twitter API
consumer_key = ""
consumer_secret = ""
access_token = ""
access_secret = ""

# Creating class listener inherited from StreamListener
class listener(StreamListener):
    def on_data(self, data):
        try:
            all_data = json.loads(data) # loading json result
            tweet = all_data["text"] # getting the value of the "text" key, which is the tweet itself
            
            sentiment_value, confidence = sentiment(tweet) # retrieving classification ('pos' or 'neg') and confidence
            if confidence >= 0.8: # getting just the tweets with confidence 80% or over
                twitter_log = open("twitter_log.txt", "a")
                twitter_log.write(sentiment_value)
                twitter_log.write('\n')
                twitter_log.close()
            print(tweet, sentiment_value, confidence)
            return True
        except:
            return True

    def on_error(self, status):
        print(status)

# Using our credentials to get the authorization and the access token
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

# Applying our credentials to retrieve the filtered tweets
twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Ireland"])
