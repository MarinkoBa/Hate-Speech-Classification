import tweepy
import os
import pandas as pd


def load_data(file_path):
    """
    Load a csv data set in a pandas dataframe.

    Parameters
    ----------
    file_path:      String
                    The path to the csv data set.

    Returns
    -------
    df:             Pandas dataframe
                    The dataframe containing the data from the csv file.
    """
    df = pd.read_csv(file_path,
                     sep=',',
                     header=None,
                     names=['tweet_id', 'label'])
    return df


def export_data(df, file_path):
    """
    Export/append a pandas dataframe as/to a csv file.

    Parameters
    ----------
    df:             DataFrame
                    The DataFrame object including tweet id, label, text and location if available.
    file_path:      String
                    The path to the output csv file.
    """
    df.to_csv(file_path,
              sep=',',
              header=None,
              index=False,
              mode='a')


def get_tweets_by_id(config, file_path):
    """
    Get the text and location from all available tweets with IDs in annotated dataset and
    export them to the file tweets.csv.

    Parameters
    ----------
    config:         AutoConfig-object
                    The object contains necessary authorization details to get
                    access to the twitter API.
    file_path:      String
                    The path to the annotated csv data set.

    """
    # load authorization details from config-file and send request to api
    auth = tweepy.OAuthHandler(config('CONSUMER_KEY'), config('CONSUMER_SECRET'))
    auth.set_access_token(config('OAUTH_TOKEN'), config('OAUTH_TOKEN_SECRET'))
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    for i in range(170):  # export data in 500-steps: 16907 / 100 = ~170
        # load annotated data set with read_data()
        df = load_data(file_path)[i * 100:(i + 1) * 100]
        df['text'] = df.tweet_id.apply(lambda tweet_id: get_tweet_text(tweet_id, api))
        df['location'] = df.tweet_id.apply(lambda tweet_id: get_tweet_location(tweet_id, api))
        export_data(df, os.path.join('data', 'tweets.csv'))


def get_tweet_text(tweet_id, api):
    try:
        return api.get_status(tweet_id).text  # return the text from the tweet with ID if tweet available
    except:
        return None


def get_tweet_location(tweet_id, api):
    try:
        lat_long = api.get_status(tweet_id).geo['coordinates']
        return str(lat_long[0]) + "|" + str(lat_long[1])  # return the location from the tweet with ID if tweet and tweet location available
    except:
        return None
