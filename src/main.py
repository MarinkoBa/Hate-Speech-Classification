import os
import pandas as pd
from decouple import config
from src.utils.get_data import get_tweets_by_id
from src.utils.get_data import load_data
from src.utils.get_data import get_datasets
from src.utils.get_data import concatenate_datasets



if __name__ == "__main__":
    # load dataset from https://github.com/zeerakw/hatespeech
    df = get_tweets_by_id(config,
                          os.path.join('data', 'NAACL_SRW_2016.csv'))
    
    # load datasets from
    #  https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data (df2)
    #  and https://github.com/jaeyk/intersectional-bias-in-ml (df3)
    df2, df3 = get_datasets(os.path.join('data', 'labeled_data.csv'),
                            os.path.join('data',
                                         'hatespeech_text_label_vote_RESTRICTED_100K.csv'))
    
    df_concatenated = concatenate_datasets(os.path.join('data', 'tweets.csv'),
                                           df2,
                                           df3)
    