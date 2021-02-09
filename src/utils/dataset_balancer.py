import pandas as pd


def balance_data(x, y):
    """
    Balances data through under-sampling. Selects so many random normal tweets as there are hate speech tweets.

    Parameters
    ----------
    x:        Pandas dataframe
              The dataframe containing all x values -> preprocessed, shape (n, 1).
    y:		  Pandas dataframe
              The dataframe containing all y values -> hate_speech, shape (n, 1).

    Returns
    ----------
    x:        Pandas dataframe
              The dataframe containing balanced x values -> preprocessed, shape (n, 1).
    y:		  Pandas dataframe
              The dataframe containing balanced y values -> hate_speech, shape (n, 1).

    """

    normal_tweets = y.loc[y['hate_speech'] == 0]
    hate_speech_tweets = y.loc[y['hate_speech'] == 1]

    hate_speech_tweets_size = hate_speech_tweets.size

    normal_tweets_random_balanced = normal_tweets.sample(n=hate_speech_tweets_size)

    y_balanced_list = [normal_tweets_random_balanced, hate_speech_tweets]
    y_balanced = pd.concat(y_balanced_list)

    y_balanced_shuffled = y_balanced.sample(frac=1)  # shuffle the whole set

    y_balanced_ind = y_balanced_shuffled.index

    x_balanced_shuffled = x.loc[y_balanced_ind]

    return x_balanced_shuffled, y_balanced_shuffled
