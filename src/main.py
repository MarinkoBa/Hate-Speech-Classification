import os
from src.utils import pipeline

if __name__ == "__main__":
    # loads and concatenates the different datasets
    df_dataset = pipeline.load_labeled_dataset()

    # evaluate classifier and parameters
    param, x_data, y_data = pipeline.run_experiment(df_dataset)

    print(param)
    pipeline.run_pipeline(os.path.join('data', 'usa_tweets.csv'), classifier=param[0], vectorizer=param[1], ngrams=param[2],
                 x_train=x_data, y_train=y_data, preprocessing='preprocessing_restricted')
    # # load dataset from https://github.com/zeerakw/hatespeech
    # df = get_tweets_by_id(config,
    #                       #os.path.join('data', 'NAACL_SRW_2016.csv'))
    #
    # # load datasets from
    # #  https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data (df2)
    # #  and https://github.com/jaeyk/intersectional-bias-in-ml (df3)
    # df2, df3 = get_datasets(os.path.join('data', 'labeled_data.csv'),
    #                             os.path.join('data',
    #                                          'hatespeech_text_label_vote_RESTRICTED_100K.csv'))
    #
    # df_concatenated = concatenate_datasets(os.path.join('data', 'tweets.csv'),
    #                                            df2,
    #                                            df3)
    #
    # # add new column with preprocessed text
    # #df_concatenated['preprocessed'] = df_concatenated['text'].apply(preprocessing)
    # # other opportunity: use restricted_preprocessing Method.
    # df_concatenated['preprocessed'] = df_concatenated['text'].apply(preprocessing_restricted)
    #
    # # balance data -> ~9k normal vs ~9k hate speech tweets
    # x_balanced, y_balanced = dataset_balancer.balance_data(df_concatenated[['preprocessed']], df_concatenated[['hate_speech']])
    #
    # # cross validation || all 4 tests include preprocessing_restricted & balanced dataset
    # # cross_validator.cross_validate(x_balanced, y_balanced, method="count",ngrams=(1,1)) # option 1 -> CountVectorizer + unigrams
    # # cross_validator.cross_validate(x_balanced, y_balanced, method="count",ngrams=(1,2)) # option 2 -> CountVectorizer + unigrams & bigrams
    # # cross_validator.cross_validate(x_balanced, y_balanced, method="tfidf",ngrams=(1,1)) # option 3 -> TfidfVectorizer + unigrams
    # # cross_validator.cross_validate(x_balanced, y_balanced, method="tfidf",ngrams=(1,2)) # option 4 -> TfidfVectorizer + unigrams & bigrams
    #
    # # Begin USA dataset
    # df_usa = pd.read_csv(os.path.join('data', 'usa_tweets.csv'),
    #                      sep=',')
    # df_usa['preprocessed'] = df_usa['full_text'].apply(preprocessing_restricted)
    #
    # # Save Model and Vectorizer
    # # svm_model, vec, x_test = svm_classifier.setup_svm_classifier(x_balanced, y_balanced, df_usa[['preprocessed']], method="tfidf",ngrams=(1,2))
    # # y_pred_svm = svm_classifier.predict(svm_model, x_test)
    # # manage_classifiers.save_classifier(svm_model, vec)
    #
    # # Load classifier
    # svm_model, vec = manage_classifiers.load_classifier(os.path.join(os.path.pardir, 'src', 'models', 'model.pkl'), os.path.join(os.path.pardir, 'src', 'models', 'vec.pkl'))
    # x_test = vec.transform(df_usa['preprocessed'].values)
    # y_pred_svm = svm_model.predict(x_test)
    #
    # # Calculate avg hate speech per US state
    # avg_hate_speech_per_state = usa_hate_speech_calculator.calculate_hate_speech_ratio(df_usa, y_pred_svm)
    #
    # unique_city_names = df_usa.city_name.unique()
    #
    # # Print avg hate speech rate per US state
    # for state in range(len(unique_city_names)):
    #     print('US state: ' + unique_city_names[state] + ' | hate speech rate: ' + str(np.round(avg_hate_speech_per_state[state], 2)) + '%')
