import os
import pandas as pd
import numpy as np
from decouple import config
from src.utils.get_data import get_tweets_by_id
from src.utils.get_data import load_data
from src.utils.get_data import get_datasets
from src.utils.get_data import concatenate_datasets
from src.utils.get_data import split_data
from src.utils.preprocessing import preprocessing
from src.utils.preprocessing import preprocessing_restricted
from src.classifiers import svm_classifier
from src.utils import cross_validator
from src.utils import usa_hate_speech_calculator
from src.utils import manage_classifiers
from src.utils import dataset_balancer


def run_pipeline():

    if not os.path.isfile(os.path.join('data', 'tweets.csv')):
        # load dataset from https://github.com/zeerakw/hatespeech, loads tweets via tweet id
        df = get_tweets_by_id(config, os.path.join('data', 'NAACL_SRW_2016.csv'))


