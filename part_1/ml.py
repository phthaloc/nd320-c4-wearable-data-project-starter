"""
Helper module for machine learning related functions
"""


import glob
import itertools
import numpy as np
import pandas as pd
import scipy as sp
import etl
import sklearn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor


__author__ = 'Udo Dehm, udacity'
__credits__ = ['Udo Dehm', 'udacity']
__license__ = ''
__version__ = '0.1.0'
__maintainer__ = 'Udo Dehm'
__email__ = 'udo.dehm@mailbox.org'
__status__ = 'Dev'


def AggregateErrorMetric(pr_errors, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability. 

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding 
            reference heart rates in BPM.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        the MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estimates = pr_errors[confidence_est >= percentile90_confidence]

    # Return the mean absolute error
    return np.mean(np.abs(best_estimates))


def confidence(freqs, fft, y_pred, confidence_freq_range, freq_range):
    """
    Compute the confidence of a frequency peak/resting pulse rate prediction
    (y_pred) given an underlying frequency power signal.
    The confidence is computed as the ratio between the integrated 
    (summed) power spectrum (fft) around (confidence_freq_range) the prediction
    (y_pred) and the integrated (summed) power spectrum of the complete signal.
    The complete signal is defined as the frequency range given by argument 
    freq_range.
    :param freqs: frequencies corresponding to the fft power signal (can be
        stacked signals)
    :param ftt: FFT power signal (can be stacked signals)
    :param y_pred: numpy array containing frequencies. These frequencies
        are predictions of the resting heart rate.
    :param confidence_freq_range: float number defining the interval size
        that is used for computing the integrated power signals for the 
        heart rate prediction. Must be smaller than the complete frequency
        interval (confidence_freq_range<(freq_range[1]-freq_range[0]))
    :param freq_range: tuple with minimum and maximum frequency defining
        the range of the complete signal.
    :return: numpy array with confidence estimates
    """
    # for computing the confidence we filter for frequencies
    # in the range of interest.
    freqs_filtered, fft_filtered = etl.filter_frequencies(
        freqs=freqs,
        fft=fft,
        freq_range=freq_range
    )

    # compute confidence:
    conf = []
    for row_freqs, row_fft, pred in zip(freqs_filtered, fft_filtered, y_pred):
        conf += [
            etl.fractional_spectral_energy(
                freqs=row_freqs,
                fft=row_fft,
                freq_range=(pred/60-0.5*confidence_freq_range,
                            pred/60+0.5*confidence_freq_range)
            )[0]
        ]
    return np.stack(conf)


def hyperparam_selection(features, labels, groups, n_estimators_opt, max_tree_depth_opt):
    """
    Train a random Forest Regression Algorithm to get the best hyperparameters 
    (n_estimators, max_tree_depth) and an idea of the performance of the parameter
    on the given dataset (defined by arguments features, labels, groups).
    In detail, we perform a nested cross validation:
    To get an accurate idea of the performance of the ML model (Random Forest Regressor),
    we pick the best hyperparameters (based on a validation set) on a subset of the data
    and evaluate it on a hold-out-set (test set). This is similar to 
    train-validation-test set split. Since the dataset is too small to separate
    it into 3 parts, we nest the hyperparameter selection in another
    layer of cross validation.
    :param features: all features the Random Forest Regressor uses for training
    :param labels: labels for each data point in features np.array
    :param groups: group each data point in features np.array belongs to
    :param n_estimators_opt: list of Random Forest hyperparameters n_estimator to use
        for training. The best n_estimator value for each test set will be safed in 
        the output DataFrame.
    :param max_tree_depth_opt: list of Random Forest hyperparameters max_tree_depth to
        use for training. The best max_tree_depth value for each test set will be safed
        in the output DataFrame.
    :return: pandas DataFrame with list of parameters and validation and test set metrics
    """
    # initialize cross validation:
    # 1 group is left out for validation or testing, the rest is used for training
    logo = LeaveOneGroupOut()
    # counter for printing iterations:
    splits = 0
    df_best_hparams = pd.DataFrame()
    # iterate over each cross-validation fold (in the first
    # iteration the indices of the first group are assigned to test_ind,
    # and the remaining indices are asigned to train_valid_ind, in the second
    # iteration the indices of the second group are assigned to test_ind,
    # and so on):
    nr_data_splits = len(list(logo.split(X=features, y=labels, groups=groups)))
    for train_valid_ind, test_ind in logo.split(X=features, y=labels, groups=groups):
        # split the dataset into a combined training and validation dataset and a
        # test set in this cross-validation fold (the training set has data from
        # only 1 group):
        # training and validation dataset:
        X_train_valid, y_train_valid = features[train_valid_ind], labels[train_valid_ind]
        # testing dataset:
        X_test, y_test = features[test_ind], labels[test_ind]
        # compute new groups array for next (nested) cross validation
        # (training / validation split):
        groups_train_valid = groups[train_valid_ind]


        ## Model selection on validation set:
        # storage for all evaluation metrics of all hyperparameter sets:
        valid_eval_metrics = []

        hparams_cross_product = itertools.product(n_estimators_opt, max_tree_depth_opt)
        for n_estimators, max_tree_depth in hparams_cross_product:
            # define regressor (ml model) for each set of hyperparameters:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_tree_depth,
                random_state=42,
                n_jobs=8
            )
            # initialize lists to collect information for evaluation metrics
            valid_score = []
            abs_valid_error = []
            # iterate over each cross-validation fold (in the first
            # iteration the indices of the first group are assigned to valid_ind,
            # and the remaining indices are asigned to train_ind, in the second
            # iteration the indices of the second group are assigned to valid_ind,
            # and so on):
            for train_ind, valid_ind in logo.split(X=X_train_valid, y=y_train_valid, groups=groups_train_valid):
                # split the dataset into training and validation dataset in 
                # this cross-validation fold (the validation set has data from
                # only 1 group):
                X_train, y_train = features[train_ind], labels[train_ind]
                X_valid, y_valid = features[valid_ind], labels[valid_ind]

                # train machine learning model
                # by calling the fit method, the model gets fitted to the current train data
                # if the model has been trained before (on a different train data) its saved
                # state (weights/decision paths/etc.) is overwritten:
                model.fit(X_train, y_train)

                # run ml model on the current validation set:
                y_pred_valid = model.predict(X_valid)

                # compute (rolling) scores for model evaluation
                valid_score += [model.score(X=X_valid, y=y_valid)]
                # compute (rolling) basis for calculating MSE and MAE for validation dataset:
                abs_valid_error += [np.abs(y_valid - y_pred_valid)]

            # transform validation scores to numpy array:
            valid_score = np.array(valid_score)
            # compute MAE and MSE for all validation sets with this hyperparameter sets
            # we compute all validation set based metrics over all validation sets (mean, std)
            # so that we get a metric that tells us what hyperparameters work best on the 
            # total dataset that is used for training and validation:
            abs_valid_error = np.concatenate(abs_valid_error)
            mae_valid = np.mean(abs_valid_error)
            mse_valid = np.mean(np.square(abs_valid_error))

            # wirte all evaluation metrics (for each hyperparameter set) to a summary list:
            valid_eval_metrics += [(
                n_estimators, max_tree_depth, mse_valid,
                mae_valid, valid_score.mean(),
                valid_score.std()
            )]

        # transform evaluation metric table to pd dataframe:
        df_valid_eval_metrics = pd.DataFrame(
            valid_eval_metrics,
            columns=['n_estimators', 'max_tree_depth', 'valid_mse',
                     'valid_mae', 'valid_score_mean', 'valid_score_std'])

        # create model with best pair of hyperparameters for this training and validation set:
        ser_best_hparams = df_valid_eval_metrics.loc[df_valid_eval_metrics['valid_mae'].argmin()].copy(deep=True)
        # train a new model on the test set with the best hyperparameter estimates
        best_model = RandomForestRegressor(
            n_estimators=int(ser_best_hparams['n_estimators']),
            max_depth=int(ser_best_hparams['max_tree_depth']),
            random_state=42,
            n_jobs=8
        )
        # fit model to the complete training + validation set
        best_model.fit(X_train_valid, y_train_valid)
        y_pred_test = best_model.predict(X_test)
        mse_test = sklearn.metrics.mean_squared_error(y_true=y_test, y_pred=y_pred_test)
        mae_test = sklearn.metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred_test)
        ser_best_hparams['test_mse'] = mse_test
        ser_best_hparams['test_mae'] = mae_test
        df_best_hparams = df_best_hparams.append(ser_best_hparams)

        splits += 1
        print(f'it. {splits}/{nr_data_splits}: MAE test set: {mae_test}')

    # reset index of dataframe:
    df_best_hparams = df_best_hparams.reset_index(drop=True)
    return df_best_hparams


def train_best_rpr_algorithm(features, labels, df_hparams=None, max_tree_depth=None, n_estimators=100):
    """
    Train a random forest regressor model on the input features and labels based on the 
    hyperparameters max_tree_depth (tree depth) and n_estimators (# of trees). The best
    estimators are selected by value counts of the hyperparameter DataFrame.
    :param features: features on which the RandomForestRegressor model should be trained
    :param labels: labels corresponding to the features
    :param df_hparams: pandas DataFrame containing hyperparameters and validation/ test
        set evaluation metrics of previously trained RandomForestRegressor models. The
        DataFrame must include columns 'max_tree_depth' and 'n_estimators'.
        If a hyperparameter dataframe is given than this has priority vor the 
        max_tree_depth and n_estimators argument.
    :param max_tree_depth: maximum depth of trees in random forest model.
    :param n_estimators: number of trees in random forest (default: 100)
    :return: trained RandomForestRegressor model
    """
    best_n_estimators = n_estimators

    if max_tree_depth is not None:
        best_max_tree_depth = max_tree_depth 
    else:
        best_max_tree_depth = None
    
    if df_hparams is not None:
        best_max_tree_depth = df_hparams['max_tree_depth'].value_counts().sort_values(ascending=False)
        best_max_tree_depth = int(best_max_tree_depth.index[0])
        
        best_n_estimators = df_hparams['n_estimators'].value_counts().sort_values(ascending=False)
        best_n_estimators = int(best_n_estimators.index[0])
    

    model = RandomForestRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_tree_depth,
        random_state=42,
        n_jobs=8
    )
    model.fit(features, labels)
    return model
