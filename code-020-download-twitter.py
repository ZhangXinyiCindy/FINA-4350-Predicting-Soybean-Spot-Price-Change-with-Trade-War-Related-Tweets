# This script listens to the Twitter stream. Keep in mind that it
# sometimes might crash if there are some weird tweets. This is most
# likely a bug in Tweepy. In any case, this means that sometimes you
# might need to restart this script. You can run this script in the
# background by typing at the Unix shell the following command (you
# can log out afterwards and it will still keep running). The command
# will create another file called `nohup.out` where all status and/or
# error messages are saved for later inspection.
#
# nohup python3 code-020-download-twitter.py &
import tweepy

# Here you need to specify your Twitter API login details. You can get
# the login details from the Twitter website.
#
# Here I read this API login information from another file, but for
# your purposes, you can specify them directly as in the commented-out
# code below.
exec(open('../code-API-key-tweepy.py').read())
# access_token = "..."
# access_token_secret = "..."
# consumer_key = "..."
# consumer_secret = "..."

# Here we prepare the Twitter API.
auth = tweepy.OAuthHandler(access_token, access_token_secret)
auth.set_access_token(consumer_key, consumer_secret)
api = tweepy.API(auth)

# Define custom listener class that is a derived class from the
# `tweepy.StreamListener` base class. All it does it to write specific
# data fields from the Twitter stream to a file. It also optionally
# checks whether a user has identified English ("en") as his main
# language. (This part is optional and currently turned off as it does
# not guarantee that the user writes all tweets in English.) More
# information about the Twitter fields can be found here:
# https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/user-object
class listener(tweepy.StreamListener):
    def on_status(self, status):
        with open('data-streaming-tweets.txt', 'a') as f:
            # if status.user.lang == 'en': # Optional check on user's language.
            f.write(            # Write the data to file.
                status.user.screen_name + ' : ' + \
                str(status.user.followers_count) + ' : ' + \
                str(status.created_at) + ' : ' + \
                status.text + '\n')
    
    def on_error(self, status_code):
        print(status_code)

# Instantiate an object of class `tweepy.Stream`.
mystream = \
    tweepy.Stream(
        auth=api.auth,
        listener=listener())
# Filter the stream for whatever you want. In this case we are
# listening for tweets that mention a select stock index/ETF, fixed
# income index/ETF, or a commodities index/ETF. More tickers could be
# added here.
mystream.filter(
    track=[
        '$SPX', '$SPY', '$ES',
        '$DJI', '$DJIA', '$INDU', '$YM',
        '$NQ', '$NASDAQ', '$QQQ',
        '$TLT',
        '$GC', '$GLD',
        '$NG', '$WTI'])
