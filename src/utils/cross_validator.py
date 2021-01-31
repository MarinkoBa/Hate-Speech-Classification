from sklearn.model_selection import KFold
from src.classifiers import svm_classifier
from src.classifiers import decision_tree_classifier
from src.classifiers import random_forest_classifier
from src.classifiers import logistic_regression
import numpy as np


def cross_validate(x, y, n_splits=10):
    """
    Calculate cross validation average error.

    Parameters
    ----------
    x:        Pandas dataframe
              The dataframe containing all x values -> preprocessed, shape (n, 1).
    y:		  Pandas dataframe
              The dataframe containing all y values -> hate_speech, shape (n, 1).

    """

    tot_err_svm = []
    tot_err_dec_tree = []
    tot_err_ran_for = []
    tot_err_log_reg = []

    kf = KFold(n_splits)
    for i, (train, test) in enumerate(kf.split(x)):
        x_train = x.iloc[train]
        y_train = y.iloc[train]
        x_test = x.iloc[train]
        y_test = y.iloc[train]


        model_svm, vec_svm, X_testing_svm  = svm_classifier.setup_svm_classifier(x_train, y_train, x_test)
        y_pred_svm = svm_classifier.predict(model_svm, X_testing_svm)
        error_svm = np.count_nonzero(y_pred_svm != np.array(y_test).flatten()) / float(len(test))
        print('Error SVM i=' + str(i) + ': ' + str(error_svm))

        model_dec_tree, vec_dec_tree, X_testing_dec_tree = decision_tree_classifier.setup_decision_tree_classifier(x_train, y_train, x_test)
        y_pred_dec_tree = decision_tree_classifier.predict(model_dec_tree, X_testing_dec_tree)
        error_dec_tree = np.count_nonzero(y_pred_dec_tree != np.array(y_test).flatten()) / float(len(test))
        print('Error Decision Tree i=' + str(i) + ': ' + str(error_dec_tree))

        model_ran_for, vec_ran_for, X_testing_ran_for = random_forest_classifier.setup_random_forest_classifier(x_train, y_train, x_test)
        y_pred_ran_for = random_forest_classifier.predict(model_ran_for, X_testing_ran_for)
        error_ran_for = np.count_nonzero(y_pred_ran_for != np.array(y_test).flatten()) / float(len(test))
        print('Error Random Forrest i=' + str(i) + ': ' + str(error_ran_for))

        model_log_reg, vec_log_reg, X_testing_log_reg = logistic_regression.setup_log_reg_classifier(x_train, y_train, x_test)
        y_pred_log_reg = logistic_regression.predict(model_log_reg, X_testing_log_reg)
        error_log_reg = np.count_nonzero(y_pred_log_reg != np.array(y_test).flatten()) / float(len(test))
        print('Error Logistic Regression i=' + str(i) + ': ' + str(error_log_reg))

        print('----------------------')

        tot_err_svm.append(error_svm)
        tot_err_dec_tree.append(error_dec_tree)
        tot_err_ran_for.append(error_ran_for)
        tot_err_log_reg.append(error_log_reg)

    print('----------------------')
    print('----------------------')
    print('Avg error SVM: ' + str(np.average(tot_err_svm)))
    print('Avg error Decision Tree: ' + str(np.average(tot_err_dec_tree)))
    print('Avg error Random Forrest: ' + str(np.average(tot_err_ran_for)))
    print('Avg error Logistic Regression: ' + str(np.average(tot_err_log_reg)))


