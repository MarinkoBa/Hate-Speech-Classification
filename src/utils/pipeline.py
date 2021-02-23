import os
import pandas as pd
import numpy as np
import constant
from decouple import config
from src.utils.get_data import get_tweets_by_id
from src.utils.get_data import load_data
from src.utils.get_data import get_datasets
from src.utils.get_data import concatenate_datasets
from src.utils.get_data import split_data
from src.utils.get_data import get_datasets
from src.utils.preprocessing import preprocessing
from src.utils.preprocessing import preprocessing_restricted
from src.classifiers import svm_classifier
from src.utils import cross_validator
from src.utils import usa_hate_speech_calculator
from src.utils import manage_classifiers
from src.utils import dataset_balancer


def run_pipeline(preprocessing = 'preprocessing_restricted'):
    # loads and concatenates the different datasets
    df_dataset = load_data()

    # runs chosen preprocess-method on the text-column of the dataframe
    if preprocessing is constant.PREPROCESSING_RESTRICTED:
        df_dataset['preprocessed'] = df_dataset['text'].apply(preprocessing_restricted)
    else:
        df_dataset['preprocessed'] = df_dataset['text'].apply(preprocessing)


    # balance data -> ~9k normal vs ~9k hate speech tweets
    x_balanced, y_balanced = dataset_balancer.balance_data(df_dataset[['preprocessed']], df_dataset[['hate_speech']])



