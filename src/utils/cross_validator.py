from sklearn.model_selection import KFold
from src.classifiers import svm_classifier
from src.classifiers import decision_tree_classifier
from src.classifiers import random_forest_classifier
from src.classifiers import logistic_regression
from src.classifiers.define_features import define_features_tfidf
from src.classifiers.ensemble_classifier import EnsembleClassifier
from sklearn import metrics
import numpy as np


def cross_validate(x, y, method ,ngrams, n_splits=10):
    """
    Calculate cross validation average error.

    Parameters
    ----------
    x:        Pandas dataframe
              The dataframe containing all x values -> preprocessed, shape (n, 1).
    y:		  Pandas dataframe
              The dataframe containing all y values -> hate_speech, shape (n, 1).
    n_splits: Integer
              Number of folds

    """

    tot_err_svm = []
    tot_err_dec_tree = []
    tot_err_ran_for = []
    tot_err_log_reg = []
    tot_err_ens = []

    tot_acc_svm = []
    tot_acc_dec_tree = []
    tot_acc_ran_for = []
    tot_acc_log_reg = []
    tot_acc_ens = []

    tot_prec_svm = []
    tot_prec_dec_tree = []
    tot_prec_ran_for = []
    tot_prec_log_reg = []
    tot_prec_ens = []

    tot_rec_svm = []
    tot_rec_dec_tree = []
    tot_rec_ran_for = []
    tot_rec_log_reg = []
    tot_rec_ens = []

    kf = KFold(n_splits)
    for i, (train, test) in enumerate(kf.split(x)):
        x_train = x.iloc[train]
        y_train = y.iloc[train]
        x_test = x.iloc[test]
        y_test = y.iloc[test]

        print('Iteration: ' + str(i))
        print('')
        print('SVM: ')
        model_svm, vec_svm, X_testing_svm = svm_classifier.setup_svm_classifier(x_train, y_train, x_test, method=method,ngrams=ngrams)
        y_pred_svm = svm_classifier.predict(model_svm, X_testing_svm)
        error_svm, acc_svm, prec_svm, rec_svm = calculate_metrics(y_test, y_pred_svm)
        print('Error: ' + str(error_svm))
        print('Accuracy: ' + str(acc_svm))
        print('Precision: ' + str(prec_svm))
        print('Recall: ' + str(rec_svm))
        print('')

        print('Decision Tree: ')
        model_dec_tree, vec_dec_tree, X_testing_dec_tree = decision_tree_classifier.setup_decision_tree_classifier(
            x_train, y_train, x_test, method=method,ngrams=ngrams)
        y_pred_dec_tree = decision_tree_classifier.predict(model_dec_tree, X_testing_dec_tree)
        error_dec_tree, acc_dec_tree, prec_dec_tree, rec_dec_tree = calculate_metrics(y_test, y_pred_dec_tree)
        print('Error: ' + str(error_dec_tree))
        print('Accuracy: ' + str(acc_dec_tree))
        print('Precision: ' + str(prec_dec_tree))
        print('Recall: ' + str(rec_dec_tree))
        print('')

        print('Random Forrest: ')
        model_ran_for, vec_ran_for, X_testing_ran_for = random_forest_classifier.setup_random_forest_classifier(x_train,
                                                                                                                y_train,
                                                                                                                x_test,
                                                                                                                method=method,
                                                                                                                ngrams=ngrams)
        y_pred_ran_for = random_forest_classifier.predict(model_ran_for, X_testing_ran_for)
        error_ran_for, acc_ran_for, prec_ran_for, rec_ran_for = calculate_metrics(y_test, y_pred_ran_for)
        print('Error: ' + str(error_ran_for))
        print('Accuracy: ' + str(acc_ran_for))
        print('Precision: ' + str(prec_ran_for))
        print('Recall: ' + str(rec_ran_for))
        print('')

        print('Logistic Regression: ')
        model_log_reg, vec_log_reg, X_testing_log_reg = logistic_regression.setup_log_reg_classifier(x_train, y_train,
                                                                                                     x_test, method=method,ngrams=ngrams)
        y_pred_log_reg = logistic_regression.predict(model_log_reg, X_testing_log_reg)
        error_log_reg, acc_log_reg, prec_log_reg, rec_log_reg = calculate_metrics(y_test, y_pred_log_reg)
        print('Error: ' + str(error_log_reg))
        print('Accuracy: ' + str(acc_log_reg))
        print('Precision: ' + str(prec_log_reg))
        print('Recall: ' + str(rec_log_reg))
        print('')

        print('Ensemble Classifier: ')
        ensemble = EnsembleClassifier()
        x_test, vec = ensemble.train(x_train, y_train, x_test, method=method,ngrams=ngrams)
        y_pred_ens = ensemble.predict(x_test)
        error_ens, acc_ens, prec_ens, rec_ens = calculate_metrics(y_test, y_pred_ens)
        print('Error: ' + str(error_ens))
        print('Accuracy: ' + str(acc_ens))
        print('Precision: ' + str(prec_ens))
        print('Recall: ' + str(rec_ens))

        print('----------------------')
        print('')
        print('')

        tot_err_svm.append(error_svm)
        tot_err_dec_tree.append(error_dec_tree)
        tot_err_ran_for.append(error_ran_for)
        tot_err_log_reg.append(error_log_reg)
        tot_err_ens.append(error_ens)

        tot_acc_svm.append(acc_svm)
        tot_acc_dec_tree.append(acc_dec_tree)
        tot_acc_ran_for.append(acc_ran_for)
        tot_acc_log_reg.append(acc_log_reg)
        tot_acc_ens.append(acc_ens)

        tot_prec_svm.append(prec_svm)
        tot_prec_dec_tree.append(prec_dec_tree)
        tot_prec_ran_for.append(prec_ran_for)
        tot_prec_log_reg.append(prec_log_reg)
        tot_prec_ens.append(prec_ens)

        tot_rec_svm.append(rec_svm)
        tot_rec_dec_tree.append(rec_dec_tree)
        tot_rec_ran_for.append(rec_ran_for)
        tot_rec_log_reg.append(rec_log_reg)
        tot_rec_ens.append(rec_ens)

    print('----------------------')
    print('----------------------')
    print('Avg error SVM: ' + str(np.average(tot_err_svm)))
    print('Avg error Decision Tree: ' + str(np.average(tot_err_dec_tree)))
    print('Avg error Random Forrest: ' + str(np.average(tot_err_ran_for)))
    print('Avg error Logistic Regression: ' + str(np.average(tot_err_log_reg)))
    print('Avg error Ensemble: ' + str(np.average(tot_err_ens)))
    print('----------------------')
    print('Avg accuracy SVM: ' + str(np.average(tot_acc_svm)))
    print('Avg accuracy Decision Tree: ' + str(np.average(tot_acc_dec_tree)))
    print('Avg accuracy Random Forrest: ' + str(np.average(tot_acc_ran_for)))
    print('Avg accuracy Logistic Regression: ' + str(np.average(tot_acc_log_reg)))
    print('Avg accuracy Ensemble: ' + str(np.average(tot_acc_ens)))
    print('----------------------')
    print('Avg precision SVM: ' + str(np.average(tot_prec_svm)))
    print('Avg precision Decision Tree: ' + str(np.average(tot_prec_dec_tree)))
    print('Avg precision Random Forrest: ' + str(np.average(tot_prec_ran_for)))
    print('Avg precision Logistic Regression: ' + str(np.average(tot_prec_log_reg)))
    print('Avg precision Ensemble: ' + str(np.average(tot_prec_ens)))
    print('----------------------')
    print('Avg recall SVM: ' + str(np.average(tot_rec_svm)))
    print('Avg recall Decision Tree: ' + str(np.average(tot_rec_dec_tree)))
    print('Avg recall Random Forrest: ' + str(np.average(tot_rec_ran_for)))
    print('Avg recall Logistic Regression: ' + str(np.average(tot_rec_log_reg)))
    print('Avg recall Ensemble: ' + str(np.average(tot_rec_ens)))
    print('----------------------')
    print('----------------------')
    f1_score_svm = 2*( (np.average(tot_prec_svm) * np.average(tot_rec_svm)) / (np.average(tot_prec_svm) + np.average(tot_rec_svm)) )
    f1_score_dec_tree = 2*( (np.average(tot_prec_dec_tree) * np.average(tot_rec_dec_tree)) / (np.average(tot_prec_dec_tree) + np.average(tot_rec_dec_tree)) )
    f1_score_ran_for = 2*( (np.average(tot_prec_ran_for) * np.average(tot_rec_ran_for)) / (np.average(tot_prec_ran_for) + np.average(tot_rec_ran_for)) )
    f1_score_log_reg = 2*( (np.average(tot_prec_log_reg) * np.average(tot_rec_log_reg)) / (np.average(tot_prec_log_reg) + np.average(tot_rec_log_reg)) )
    f1_score_ens = 2*( (np.average(tot_prec_ens) * np.average(tot_rec_ens)) / (np.average(tot_prec_ens) + np.average(tot_rec_ens)) )
    print('F1 Score SVM: ' + str(f1_score_svm))
    print('F1 Score Decision Tree: ' + str(f1_score_dec_tree))
    print('F1 Score Random Forrest: ' + str(f1_score_ran_for))
    print('F1 Score Logistic Regression: ' + str(f1_score_log_reg))
    print('F1 Score Ensemble: ' + str(f1_score_ens))




def calculate_metrics(y_test, y_pred):
    """
    Calculate metrics: error, accuracy, precision & recall.

    Parameters
    ----------
    y_test:        Pandas dataframe
                   The dataframe containing ground truth test data.
    y_pred:		   Pandas dataframe
                   The dataframe containing prediction to the test data.

    Returns
    -------
    error:		   float
    			   proportion of false predictions w.r.t all data
    accuracy:	   float
    			   proportion of correct predictions w.r.t all data
    precision:	   float
    			   proportion of correct predictions w.r.t true positives and false positives
    recall:		   float
    			   proportion of correct predictions w.r.t true positives and false negatives

    """
    error = np.count_nonzero(y_pred != np.array(y_test).flatten()) / float(len(y_test))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    return error, accuracy, precision, recall
