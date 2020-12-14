import tweepy
import pandas as pd

def load_data(file_path):
    '''
    Load a csv data set in a pandas dataframe.
    
    Parameters
    ----------
    file_path:      String
                    The path to the csv data set.

    Returns
    -------
    df:             Pandas dataframe
                    The dataframe containing the data from the csv file.              
    '''
    df = pd.read_csv(file_path,
                     sep = ',',
                     header = None,
                     names = ['tweet_id', 'label'])
    return df



def get_tweets_by_id(config, file_path):
    '''
    Get the text from all available tweets with IDs in annotated dataset and
    save it in the dataframe.
    
    Parameters
    ----------
    config:         JSON-object
                    The object contains necessary authorization details to get
                    access to the twitter API.
    file_path:      String
                    The path to the annotated csv data set.

    Returns
    -------
    df:             Pandas dataframe
                    The dataframe containing the data from the csv file and the
                    texts of the tweets if available.              
    '''
    # load authorization details from config-file and send request to api
    auth = tweepy.OAuthHandler(config['CONSUMER_KEY'], config['CONSUMER_SECRET'])
    auth.set_access_token(config['OAUTH_TOKEN'], config['OAUTH_TOKEN_SECRET'])
    api = tweepy.API(auth)
    
    # load annotated data set with read_data()
    df = load_data(file_path)
    
    ### TRY OUT, else via loop -> api.get_status(tweet_id).text
    ### FIXME, what happens if tweet is no longer available? -> count?
    df['text'] = df.tweet_id.apply(api.get_status).text
    
    ### TODO: can get information about geolocation (if available),
    #   have a look at examples when authorization works

    return df
