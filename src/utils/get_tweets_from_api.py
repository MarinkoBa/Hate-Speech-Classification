# -*- coding: utf-8 -*-
import tweepy
import pandas as pd

import json


def setup_connection(config):
    """
    Set up the authorization process to connect to the Twitter API.
    
    Parameters
    ----------
    config:         AutoConfig-object
                    The object contains necessary authorization details to get
                    access to the twitter API.
    
    Returns
    -------
    api             Twitter API wrapper
                    It provides a wrapper for the API of Twitter.
    """
    # authenticate
    auth = tweepy.OAuthHandler(config('CONSUMER_KEY'), config('CONSUMER_SECRET'))
    auth.set_access_token(config('OAUTH_TOKEN'), config('OAUTH_TOKEN_SECRET'))

    api = tweepy.API(auth,
                     wait_on_rate_limit = True,
                     wait_on_rate_limit_notify = True)
    
    return api




def get_location_tweets(api, df_all, geocode, city_name):
    """
    Get tweets by location 
    
    Parameters
    ----------
    api             Twitter API wrapper
                    It provides a wrapper for the API of Twitter.
    df_all          Pandas Dataframe
                    The dataframe containing the data from previous searches
                    with another location as parameter.
    geocode         String
                    The location given in "latitude,longitude,radius",
                    with radius units as "km" (here
    city_name       String
                    The name of the city to add as a new column for further
                    analysis and evaluation.

    Returns
    -------
    df_all          Pandas Dataframe
                    The dataframe from the input with the concatenated results
                    from this query for the specified city.
    """

    replies = []

    for tweet in tweepy.Cursor(api.search,
                               q="-is:retweet",
                               count=300,
                               geocode = geocode, # e.g. New York: "40.71427,-74.00597,10km",
                               lang = "en",
                               tweet_mode = "extended").items(300):
        replies.append(tweet)
    
    tweets_python = []
    for tweet in replies:
        #convert to string
        json_obj = json.dumps(tweet._json)
        
        # convert to dict
        parsed = json.loads(json_obj)

        tweets_python.append(parsed)
     
    df = pd.DataFrame(tweets_python)
    df = df.drop(['id_str', 'entities',
                  'truncated',
                 'metadata', 'source',
                 'in_reply_to_status_id',
                 'in_reply_to_status_id_str',
                 'in_reply_to_user_id',
                 'in_reply_to_user_id_str',
                 'in_reply_to_screen_name',
                 'is_quote_status',
                 'retweet_count',
                 'favorite_count',
                 'favorited', 'retweeted',
                 'lang', 'possibly_sensitive',
                 'quoted_status_id', 
                 'quoted_status_id_str',
                 'geo', 'coordinates',
                 'contributors', 'extended_entities',
                 'display_text_range'], axis = 1)
    
    # add column with city_name
    df = df.assign(city_name = city_name) # e.g. here: New York
    
    df_all = pd.concat([df_all, df], ignore_index = True)
    
    ### drop all which are only retweets? so if retweeted_status is not empty?
    return df_all