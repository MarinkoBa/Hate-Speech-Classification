import os
import pandas as pd
import numpy as np
from src.utils import constant
from decouple import config
from src.utils.get_data import get_tweets_by_id
from src.utils.get_data import concatenate_datasets
from src.utils.get_data import get_datasets
from src.utils.preprocessing import preprocessing_restricted
from src.utils.cross_validator import validate_parameters_via_cross_validation
from src.utils import usa_hate_speech_calculator
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


def run_experiment(df_dataset, preprocessing='preprocessing_restricted'):
    """
    Execute preprocessing on given dataset and validates best classifier and parameters by f1-score.
    Parameters
    ----------

    df_dataset          Pandas dataframe
                        Dataframe including features and labels.
                        On Dataframe Experiments will be executed.

    preprocessing       String
                        Preprocessing methode which should be used.
                        preprocessing_restricted or otherwise preprocessing methode will 
                        be used.


    Returns
    -------
    Parameters:         Array
                        including 0: classifer, 1: vectorizer, 2: grams
                        classifier: svm, decison_tree,random_forest, log_reg or ensemble
                        vectorizer: count or tfidf
                        ngram represented as Tuple e.g (1, 2) for uni- and bigrams

    x_balanced          Array
                        Balanced and preprocessed training features

    y_balanced          Array
                        balanced training labels
    """

    # runs chosen preprocess-method on the text-column of the dataframe
    if preprocessing is constant.PREPROCESSING_RESTRICTED:
        df_dataset['preprocessed'] = df_dataset['text'].apply(preprocessing_restricted)
    else:
        df_dataset['preprocessed'] = df_dataset['text'].apply(preprocessing)

    # balance data -> ~9k normal vs ~9k hate speech tweets
    x_balanced, y_balanced = dataset_balancer.balance_data(df_dataset[['preprocessed']],
                                                           df_dataset[['hate_speech']])

    # cross validation
    parameters = validate_parameters_via_cross_validation(x_data=x_balanced,
                                                          y_data=y_balanced)

    return parameters, x_balanced, y_balanced


def make_prediction(dataset_filepath, classifier, vectorizer, ngrams, x_train, y_train,
                    preprocessing='preprocessing_restricted'):
    """
    Make predictions to the input csv-file with the given classifier and parameters.

    Parameters
    ----------
    dataset_filepath:  	String.
                        Filepath to the csv-file which includes the data which should be investigate.
                        The feature column has to be named 'full_text'.

    classifier:         String
                        Names of classifier which should be used.
                        svm, decison_tree,random_forest, log_reg or ensemble

    vectorizer: 		String
                        Name of Vectorizer which should be used.
                        count or tfidf

    ngrams:             Tuple
                        Unigram: (1,1) ,Uni-/Bigrams: (1,2)

    x_train             Array.
                        Features of the training data

    y_train             Array.
                        Labels of the training data

    preprocessing       String
                        Preprocessing methode which should be used.
                        preprocessing_restricted or otherwise preprocessing methode will be used.

    Returns
    -------

    y_pred              Array
                        Predicted labels for the input dataset (csv-file)

    df                  Pandas Dataframe
                        Dataframe loaded from csv-file
    """
    # load dataset from the given filepath
    df = pd.read_csv(dataset_filepath, sep=',')

    # runs chosen preprocess-method on the full_text-column of the dataframe
    if preprocessing is constant.PREPROCESSING_RESTRICTED:
        df['preprocessed'] = df['full_text'].apply(preprocessing_restricted)
    else:
        df['preprocessed'] = df['full_text'].apply(preprocessing)

    model, vec, x_test = choose_and_create_classifier(classifier, x_train, y_train, df[['preprocessed']], vectorizer,
                                                      ngrams)
    x_test = vec.transform(df['preprocessed'].values)
    y_pred = model.predict(x_test)

    return y_pred, df


def calculate_hatespeech_per_us_state(df_usa, y_pred, print_out=True):
    """
    Make predictions to the input csv-file with the given classifier and parameters.

    Parameters
    ----------
    df_usa:  	Pandas dataframe
                Dataset of the collected tweets from different US-States

    y_pred:     Array.
                Predicted labels for the dataset features

    print_out:  Bool.
                True if each average should be print to console

    Returns
    -------

    y_pred      Array
                Predicted labels for the input dataset (csv-file)
    """
    # Calculate avg hate speech per US state
    avg_hate_speech_per_state = usa_hate_speech_calculator.calculate_hate_speech_ratio(df_usa, y_pred)
    unique_city_names = df_usa.city_name.unique()

    # Print avg hate speech rate per US state
    if print_out:
        for state in range(len(unique_city_names)):
            print('US state: ' + unique_city_names[state] + ' | hate speech rate: ' + str(
                np.round(avg_hate_speech_per_state[state], 2)) + '%')

    return avg_hate_speech_per_state
