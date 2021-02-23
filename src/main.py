import os
from src.utils import pipeline

if __name__ == "__main__":
    # loads and concatenates the different datasets
    df_dataset = pipeline.load_labeled_dataset()

    # evaluate classifier and parameters during experiment-phase
    param, x_data, y_data = pipeline.run_experiment(df_dataset)

    # predict data for the usa_tweets with the evaluated parameters
    y_pred, df = pipeline.make_prediction(os.path.join('data', 'usa_tweets.csv'), classifier=param[0],
                                          vectorizer=param[1], ngrams=param[2],
                                          x_train=x_data, y_train=y_data, preprocessing='preprocessing_restricted')
    # calculate the average hatespeech per state (in percent)
    avg_per_state = pipeline.calculate_hatespeech_per_us_state(df, y_pred)
