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

    for i in range(170):  # export data in 100-steps: 16907 / 100 = ~170
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
  
    
    
def get_datasets(first_file_path, second_file_path):
    """
    Load csv data sets in a pandas dataframe.

    Parameters
    ----------
    first_file_path:        String
                            The path to the first csv data set.
    
    second_file_path:       String
                            The path to the second csv data set.

    Returns
    -------
    df2:            Pandas dataframe
                    The dataframe containing the data from the first csv file.
                    
    df3:            Pandas dataframe
                    The dataframe containing the data from the second csv file.
    """
    # class labels: 0 - hate speech 1 - offensive language 2 - neither
    df2 = pd.read_csv(first_file_path,
                      sep = ',')
    
    # drop columns with counts of people who voted for different classes
    df2 = df2.drop(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'], axis=1)
    
    # rename class to label
    df2 = df2.rename(columns = {'class':'label',
                                'tweet' : 'text'})
     
    # class labels: 'abusive', 'hateful', 'normal', 'spam'
    df3 = pd.read_csv(second_file_path,
                      header = None,
                      names = ['text', 'label', 'votes'],
                      sep='\t')
    
    df3 = df3.drop(['votes'], axis=1)
    # altering the DataFrame 
    df3 = df3[['label', 'text']] 
    
    export_data(df2, os.path.join('data', 'tweets_labeled_data.csv'))
    export_data(df3, os.path.join('data', 'tweets_hatespeech_text.csv'))
    
    return df2, df3


def sort_to_hatespeech(x):
    if x == "racism" or x == "sexism" or x == 0 or x == "hateful":
        return 1
    else:
        return 0



def concatenate_datasets(file_path_tweets, df2, df3):
    """
    Concatenate the data sets in a pandas dataframe together,
    add a column to identify the tweet as hate speech or not.

    Parameters
    ----------
    file_path_tweets:       String
                            The path to the csv data set 'tweets.csv'.
    
    df2:                    Pandas dataframe
                            The dataframe containing the data from 
                            https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data (df2)
    
    df3:                    Pandas dataframe
                            The dataframe containing the data from 
                            https://github.com/jaeyk/intersectional-bias-in-ml (df3)
.

    Returns
    -------
    df_concatenated:        Pandas dataframe
                            The dataframe containing all data from the input dataframes.
    """
    # load first data set from data/tweets.csv
    df = pd.read_csv(file_path_tweets,
                 sep=',',
                 header=None,
                 names=['label', 'text', 'FIXME'])
    
    # drop column FIXME -> why is it even there?, then drop rows without text
    df = df.drop(['FIXME'], axis=1)
    df = df.dropna()
    
    # concatenate alle dataframes
    df_concatenated = pd.concat([df, df2, df3],
                                ignore_index = True)
    
    # drop columns with labels 1 ('offensive language'), 'abusive' and 'spam'
    df_concatenated = df_concatenated[df_concatenated.label != 1]
    df_concatenated = df_concatenated[df_concatenated.label != 'abusive']
    df_concatenated = df_concatenated[df_concatenated.label != 'spam']
    
    # add column 'hate_speech' with 1 for hate speech, 0 for no hate speech/ normal
    df_concatenated['hate_speech'] = df_concatenated['label'].apply(sort_to_hatespeech)
    
    return df_concatenated

