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
from src.classifiers.ensemble_classifier import EnsembleClassifier
from src.utils import cross_validator
from src.utils import usa_hate_speech_calculator
from src.utils import manage_classifiers
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

    # balance data -> ~9k normal vs ~9k hate speech tweets
    x_balanced, y_balanced = dataset_balancer.balance_data(df_concatenated[['preprocessed']], df_concatenated[['hate_speech']])

    # cross validation || all 4 tests include preprocessing_restricted & balanced dataset
    # cross_validator.cross_validate(x_balanced, y_balanced, method="count",ngrams=(1,1)) # option 1 -> CountVectorizer + unigrams
    # cross_validator.cross_validate(x_balanced, y_balanced, method="count",ngrams=(1,2)) # option 2 -> CountVectorizer + unigrams & bigrams
    # cross_validator.cross_validate(x_balanced, y_balanced, method="tfidf",ngrams=(1,1)) # option 3 -> TfidfVectorizer + unigrams
    # cross_validator.cross_validate(x_balanced, y_balanced, method="tfidf",ngrams=(1,2)) # option 4 -> TfidfVectorizer + unigrams & bigrams

    # Begin USA dataset
    df_usa = pd.read_csv(os.path.join('data', 'usa_tweets.csv'),
                         sep=',')
    df_usa['preprocessed'] = df_usa['full_text'].apply(preprocessing_restricted)

    # Save Model and Vectorizer
    #ensemble = EnsembleClassifier()
    #x_test, vec = ensemble.train(x_balanced, y_balanced, df_usa[['preprocessed']], method="tfidf", ngrams=(1, 1))
    #y_pred_ens = ensemble.predict(x_test)
    #manage_classifiers.save_classifier(ensemble, vec)

    # Load classifier
    ensemble, vec = manage_classifiers.load_classifier(r'C:\Users\marin\Desktop\Hate-Speech-Classification\src\models\model.pkl', r'C:\Users\marin\Desktop\Hate-Speech-Classification\src\models\vec.pkl')
    x_test = vec.transform(df_usa['preprocessed'].values)
    y_pred_ens = ensemble.predict(x_test)

    # Calculate avg hate speech per US state
    avg_hate_speech_per_state = usa_hate_speech_calculator.calculate_hate_speech_ratio(df_usa, y_pred_ens)

    unique_city_names = df_usa.city_name.unique()

    # Print avg hate speech rate per US state
    for state in range(len(unique_city_names)):
        print('US state: ' + unique_city_names[state] + ' | hate speech rate: ' + str(np.round(avg_hate_speech_per_state[state], 2)) + '%')