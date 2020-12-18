import os
import pandas as pd
from decouple import config
from src.utils.get_tweets_by_id import get_tweets_by_id


if __name__ == "__main__":

    df = get_tweets_by_id(config,
                          os.path.join('data', 'NAACL_SRW_2016.csv'))
    
    
    # class labels: 0 - hate speech 1 - offensive language 2 - neither
    df2 = pd.read_csv(os.path.join('data', 'labeled_data.csv'),
                  sep = ',')

    # drop columns with counts of people who voted for different classes
    df2 = df2.drop(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'], axis=1)
    
    # rename class to label
    df2 = df2.rename(columns = {'class':'label',
                                'tweet' : 'text'})
     
    
    # class labels: 'abusive', 'hateful', 'normal', 'spam'
    df3 = pd.read_csv(os.path.join('data', 'hatespeech_text_label_vote_RESTRICTED_100K.csv'),
                      header = None,
                      names = ['text', 'label', 'votes'],
                      sep='\t')
    
    df3 = df3.drop(['votes'], axis=1)
    # altering the DataFrame 
    df3 = df3[['label', 'text']] 