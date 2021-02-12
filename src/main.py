import os
import pandas as pd
from decouple import config
from src.utils.get_data import get_tweets_by_id
from src.utils.get_data import load_data
from src.utils.get_data import get_datasets
from src.utils.get_data import concatenate_datasets
from src.utils.get_data import split_data
from src.utils.preprocessing import preprocessing
from src.utils.preprocessing import preprocessing_restricted
from src.utils import cross_validator
from src.utils import dataset_balancer



if __name__ == "__main__":
    # load dataset from https://github.com/zeerakw/hatespeech
    #df = get_tweets_by_id(config,
                          #os.path.join('data', 'NAACL_SRW_2016.csv'))

    # load datasets from
    #  https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data (df2)
    #  and https://github.com/jaeyk/intersectional-bias-in-ml (df3)
    df2, df3 = get_datasets(os.path.join('data', 'labeled_data.csv'),
                            os.path.join('data',
                                         'hatespeech_text_label_vote_RESTRICTED_100K.csv'))

    df_concatenated = concatenate_datasets(os.path.join('data', 'tweets.csv'),
                                           df2,
                                           df3)

    # add new column with preprocessed text
    #df_concatenated['preprocessed'] = df_concatenated['text'].apply(preprocessing)
    # other opportunity: use restricted_preprocessing Method.
    df_concatenated['preprocessed'] = df_concatenated['text'].apply(preprocessing_restricted)


    x_balanced, y_balanced = dataset_balancer.balance_data(df_concatenated[['preprocessed']], df_concatenated[['hate_speech']])

    # cross validation
    cross_validator.cross_validate(x_balanced, y_balanced)

    # split in test and training set
    #training_data, testing_data, training_y, testing_y = split_data(df_concatenated,
    #                                                                'preprocessed',
    #                                                                'hate_speech',
    #                                                                0.25)