import os
import pandas as pd
import numpy as np
from src.utils import constant
from decouple import config
from src.utils.get_data import get_tweets_by_id
from src.utils.get_data import load_data
from src.utils.get_data import get_datasets
from src.utils.get_data import concatenate_datasets
from src.utils.get_data import split_data
from src.utils.get_data import get_datasets
from src.utils.preprocessing import preprocessing
from src.utils.preprocessing import preprocessing_restricted
from src.utils.cross_validator import validate_parameters_via_cross_validation
from src.utils import usa_hate_speech_calculator
from src.utils import manage_classifiers
from src.utils import dataset_balancer
from src.utils.manage_classifiers import choose_and_create_classifier


def load_labeled_dataset():
    """
        Concatenate the data sets from csv-files (labeled_data.csv, hatespeech_text_label_vote_RESTRICTED_100K.csv,
        tweets.csv) together and return it as a pandas dataframe.

        Returns
        -------
        df_concatenated:        Pandas dataframe
                                The dataframe containing all data from the mentioned csv-files.
        """
    # if tweets not already loaded from TwitterAPI
    if not os.path.isfile(os.path.join('data', 'tweets.csv')):
        # load dataset from https://github.com/zeerakw/hatespeech, loads tweets via tweet id
        df = get_tweets_by_id(config, os.path.join('data', 'NAACL_SRW_2016.csv'))


    # load datasets from
    #  https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data (df2)
    #  and https://github.com/jaeyk/intersectional-bias-in-ml (df3)
    df2, df3 = get_datasets(os.path.join('data', 'labeled_data.csv'),
                            os.path.join('data', 'hatespeech_text_label_vote_RESTRICTED_100K.csv'))

    df_concatenated = concatenate_datasets(os.path.join('data', 'tweets.csv'), df2, df3)

    return df_concatenated

def run_experiment(df_dataset,preprocessing='preprocessing_restricted'):
    # TODO Documentation

    # runs chosen preprocess-method on the text-column of the dataframe
    if preprocessing is constant.PREPROCESSING_RESTRICTED:
        df_dataset['preprocessed'] = df_dataset['text'].apply(preprocessing_restricted)
    else:
        df_dataset['preprocessed'] = df_dataset['text'].apply(preprocessing)

    # balance data -> ~9k normal vs ~9k hate speech tweets
    x_balanced, y_balanced = dataset_balancer.balance_data(df_dataset[['preprocessed']], df_dataset[['hate_speech']])

    # cross validation TODO change back [:1000]
    parameters = validate_parameters_via_cross_validation(x_data=x_balanced[:1000], y_data=y_balanced[:1000])

    return parameters, x_balanced, y_balanced


def run_pipeline(dataset_filepath, classifier, vectorizer, ngrams, x_train, y_train,preprocessing='preprocessing_restricted'):
    """
    TODO write Docu
        Run pipeline.

        Parameters
        ----------
        dataset_filepath:  	Filepath to the csv-file which includes the data which should be investigate.
                            The feature column has to be named 'full_text'.

        classifier:         String
                            Names of classifier which should be used.
                            svm, decison_tree,random_forest, log_reg or ensemble

        vectorizer: 		String
                			Name of Vectorizer which should be used.
                			count or tfidf

        ngrams:             Tuple
                            Unigram: (1,1) ,Bigram: (1,2)

        Returns
        -------

        """
    # load dataset from the given filepath
    df = pd.read_csv(dataset_filepath, sep=',')

    # runs chosen preprocess-method on the full_text-column of the dataframe
    if preprocessing is constant.PREPROCESSING_RESTRICTED:
        df['preprocessed'] = df['full_text'].apply(preprocessing_restricted)
    else:
        df['preprocessed'] = df['full_text'].apply(preprocessing)


    model, vec, x_test = choose_and_create_classifier(classifier,x_train, y_train, df[['preprocessed']],vectorizer,ngrams)
    x_test = vec.transform(df['preprocessed'].values)
    y_pred = model.predict(x_test)
    print(y_pred)
