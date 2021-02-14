
def calculate_hate_speech_ratio(df_usa, y_pred):
    """
    Calculates avg hate speech per US state.

    Parameters
    ----------
    df_usa:	        Pandas dataframe
                        US test dataframe
    y_pred:	        Numpy array
                        Predicted values over df_usa test set

    Returns
    ----------
    avg_hate_speech_per_state:		        Numpy array
            			Includes 50 avg hate speech values
    """

    unique_city_names = df_usa.city_name.unique()
    avg_hate_speech_per_state = []
    for i in range(len(unique_city_names)):
        indices_cur_state = df_usa.index[df_usa['city_name'] == unique_city_names[i]]
        current_state_prediction = y_pred[indices_cur_state]
        hate_speech_amount = current_state_prediction[current_state_prediction == 1]
        hate_speech_ratio = len(hate_speech_amount) / len(current_state_prediction)
        avg_hate_speech_per_state.append(hate_speech_ratio)

    return avg_hate_speech_per_state