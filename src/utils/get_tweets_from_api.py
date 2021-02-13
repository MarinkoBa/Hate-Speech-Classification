# -*- coding: utf-8 -*-
import tweepy
import pandas as pd
import numpy as np
import os
import time
import json

        
from get_data import export_data


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
    # search-query from API for tweets in english at specified location
    for tweet in tweepy.Cursor(api.search,
                               q="-is:retweet",
                               count=2000,
                               geocode = geocode, # e.g. New York: "40.71427,-74.00597,50km",
                               lang = "en",
                               tweet_mode = "extended").items(2000):
        replies.append(tweet)
    
    # convert status objects to dictionaries
    tweets_python = []
    for tweet in replies:
        #convert to string
        json_obj = json.dumps(tweet._json)
        
        # convert to dict
        parsed = json.loads(json_obj)

        tweets_python.append(parsed)
    
    # load all in pandas dataframe
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
                 'display_text_range',
                 'withheld_in_countries',
                 'quoted_status',
                 'withheld_scope'], axis = 1, 
                errors='ignore') # so only columns are dropped if they exist
    
    # add column with city_name
    df = df.assign(city_name = city_name) # e.g. here: New York
    

    # don't drop retweets, in training data set they are included as well
    # BUT: drop those with the same retweeted tweet, so that one tweet can occur at most 2 times:
    
    # fill all nan-values in retweeted_status with 0
    df.retweeted_status = df.retweeted_status.fillna(0)
    # new column for retweeted id if exists
    df["retweeted_status_id"] = df.retweeted_status.apply(get_retweeted_status_id)
    
    # first keep all tweets which aren't retweets
    df_na = df[df.retweeted_status_id.isna()]
    
    # now delete those who have the same retweeted_id, keep only one retweet
    df_withoutna = df.dropna(subset = ["retweeted_status_id"])
    df_withoutna = df_withoutna.drop_duplicates(subset=["retweeted_status_id"],
                                                keep="first",
                                                ignore_index=True)
    
    # now concatenate
    df = pd.concat([df_na, df_withoutna], ignore_index = True)
    
    
    status_file = os.path.join("src", "utils", "query_status.txt")
    
    with open(status_file, 'a+') as file:
        file.write(f"{city_name}: {df.shape[0]} tweets \n")
    
    # concatenate with overall dataframe (so with all other already queried locations)
    df_all = pd.concat([df_all, df], ignore_index = True)

    return df_all


def get_all_locations(api):
    df_all = pd.DataFrame()
    
    # list of biggest city per state in the US
    location_list = [["33.543682,-86.779633,50km", "Birmingham, Alabama"],
                     ["61.217381,-149.863129,50km", "Anchorage, Alaska"],
                     ["33.448376,-112.074036,50km", "Phoenix, Arizona"],
                     ["34.746483,-92.289597,50km", "Little Rock, Arkansas"],
                     ["34.052235,-118.243683,50km", "Los Angeles, California"],
                     ["39.742043,-104.991531,50km", "Denver, Colorado"],
                     ["41.186390,-73.195557,50km", "Bridgeport, Connecticut"],
                     ["39.739071,-75.539787,50km", "Wilmington, Delaware"],
                     ["30.332184,-81.655647,50km", "Jacksonville, Florida"],
                     ["33.753746,-84.386330,50km", "Atlanta, Georgia"],
                     ["21.315603,-157.858093,50km", "Honolulu, Hawaii"],
                     ["43.618881,-116.215019,50km", "Boise, Idaho"],
                     ["41.881832,-87.623177,50km", "Chicago, Illinois"],
                     ["39.791000,-86.148003,50km", "Indianapolis, Indiana"],
                     ["41.619549,-93.598022,50km", "Des Moines, Iowa"],
                     ["37.697948,-97.314835,50km", "Wichita, Kansas"],
                     ["38.328732,-85.764771,50km", "Louisville, Kentucky"],
                     ["29.951065,-90.071533,50km", "New Orleans, Louisiana"],
                     ["43.67825,-70.31755,50km", "Portland, Maine"],
                     ["39.299236,-76.609383,50km", "Baltimore, Maryland"],
                     ["42.361145,-71.057083,50km", "Boston, Massachusetts"],
                     ["42.331429,-83.045753,50km", "Detroit, Michigan"],
                     ["44.986656,-93.258133,50km", "Minneapolis, Minnesota"],
                     ["32.298756,-90.184807,50km", "Jackson, Mississippi"],
                     ["39.099724,-94.578331,50km", "Kansas City, Missouri"],
                     ["45.787636,-108.489304,50km", "Billings, Montana"],
                     ["41.257160,-95.995102,50km", "Omaha, Nebraska"],
                     ["36.114647,-115.172813,50km", "Las Vegas, Nevada"],
                     ["43.008663,-71.454391,50km", "Manchester, New Hamshire"],
                     ["40.735657,-74.172363,50km", "Newark, New Jersey"],
                     ["35.106766,-106.629181,50km", "Albuquerque, New Mexico"],
                     ["40.71427,-74.00597,50km", "New York City, New York"],
                     ["35.227085,-80.843124,50km", "Charlotte, North Carolina"],
                     ["46.877186,-96.789803,50km", "Fargo, North Dakota"],
                     ["39.983334,-82.983330,50km", "Columbus, Ohio"],
                     ["35.481918,-97.508469,50km", "Oklahoma City, Oklahoma"],
                     ["45.523064,-122.676483,50km", "Portland, Oregon"],
                     ["39.952583,-75.165222,50km", "Philadelphia, Pennsylvania"],
                     ["41.825226,-71.418884,50km", "Providence, Rhode Island"],
                     ["32.784618,-79.940918,50km", "Charleston, South Carolina"],
                     ["43.536388,-96.731667,50km", "Sioux Falls, South Dakota"],
                     ["36.174465,-86.767960,50km", "Nashville, Tennessee"],
                     ["29.749907,-95.358421,50km", "Houston, Texas"],
                     ["40.7607800,-111.8910500,50km", "Salt Lake City, Utah"],
                     ["44.475883,-73.212074,50km", "Burlington, Vermont"],
                     ["36.863140,-76.015778,50km", "Virginia Beach, Virginia"],
                     ["47.608013,-122.335167,50km", "Seattle, Washington"],
                     ["38.349819,-81.632622,50km", "Charleston, West Virginia"],
                     ["43.038902,-87.906471,50km", "Milwaukee, Wisconsin"],
                     ["41.161079,-104.805450,50km", "Cheyenne, Wyoming"]]
                     # source: https://www.latlong.net/place/, 09.02.21, 10:45
    
    # iterate through location list and concatenate with overall df_all
    for location in location_list:
        try:
            df_all = get_location_tweets(api = api,
                                         df_all = df_all,
                                         geocode = location[0], 
                                         city_name = location[1])
        except Exception as e:
            print(f"{e} for {location[1]}")
            
        # sleep for 5 minutes
        time.sleep(300)
    
    # export to csv file
    export_data(df_all, os.path.join("src", "data", "usa_tweets.csv"))
    return df_all



def get_retweeted_status_id(retweeted_status):
    """
    Get retweeted_status id which is in column retweeted_status 
    
    Parameters
    ----------
    retweeted_status            0 or dict
                                Retweeted status if this tweet retweets it.

    Returns
    -------
    retweeted_status_id         NaN or String
                                The retweeted status id as a string or a Nan,
                                if there is no retweet.
    """
    if retweeted_status == 0:
        return np.nan
    else:
        return str(retweeted_status["id"])













