import pandas as pd
import numpy as np
import argparse

import rfpimp
import sklearn
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint
import time
import matplotlib.pyplot as plt

from joblib import dump, load

import os
from pathlib import Path


def get_importance(args: argparse.Namespace) -> None:
    """

    Source: https://explained.ai/rf-importance/index.html

    :param args:
    :return:
    """

    recursive_feat = False
    cv_optim_feat = True
    rand_search = True
    nested_rfecv = False


    input_ = args.input

    p = Path(input_)
    p = p.parent
    p = p.parent

    importance = p / 'importance'
    model_checkpoints = p / 'model_checkpoints'
    rf_best_params = p / 'rf_best_params'
    transform_mask = p / 'transform_mask'

    if not importance.exists():
        importance.mkdir()

    if not model_checkpoints.exists():
        model_checkpoints.mkdir()

    if not rf_best_params.exists():
        rf_best_params.mkdir()

    if not transform_mask.exists():
        transform_mask.mkdir()

    df_orig = pd.read_excel(input_)

    orig = df_orig.as_matrix()[:, 1:]

    feature_names = list(df_orig.columns)[1:-1]

    whereNan = np.isnan(list(orig[:, -1]))

    olds = orig[np.logical_not(whereNan)]

    news = orig[whereNan]

    # df_nonull = df_orig.dropna()

    y_train = olds[:, -1]
    y_train = y_train.astype('int')
    X_train = olds[:, :-1]



    X_test = news[:, :-1]

    Xdf = pd.DataFrame(X_train, columns=feature_names)
    ydf = pd.Series(y_train)

    ## Recursive feature elimination based on importance and RFE method
    if recursive_feat is True:
        print("Selecting the best features in your dataset using RFE.")
        rf_elim = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, random_state=42, bootstrap=True, n_estimators=20,
                                                          max_features=0.20)

        print("The original Xdf is shape: ")
        print(X_train.shape)

        select_fm = sklearn.feature_selection.RFE(estimator=rf_elim, n_features_to_select=30, step=1)

        select_fm.fit_transform(X_train, y_train)

        feature_conds = select_fm.get_support()
        transform_df = pd.DataFrame(feature_conds)
        transform_df.to_csv(
            str(transform_mask) + "/Transform_SSOL6_RFE_fs1_" + str(time.strftime("%Y-%m-%d-%I-%M")) + ".csv")
        X_train = select_fm.transform(X_train)

        print("Finished transforming the data; new xdf shape is: ")
        print(X_train.shape)

        Xdf = Xdf[Xdf.columns[feature_conds]]



    # Determine optimal number of features and what those features are in a cross validated format.
    # Passed to gridsearch CV or next function.
    if cv_optim_feat:
        num_of_trees = 10
        max_feat_pct = 0.30
        print("Determining the optimal number of features and what they are for {} decision trees "
              "and {} of features at each split".format(num_of_trees, max_feat_pct))

        print("The original Xdf is shape: ")
        print(X_train.shape)

        rf_base = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, random_state=42, bootstrap=True,
                                                         n_estimators=num_of_trees, max_features=max_feat_pct)

        rf_rfecv = sklearn.feature_selection.RFECV(estimator=rf_base, step=1, cv=5, n_jobs=-1,
                                                   min_features_to_select=10,
                                                   scoring='accuracy', verbose=1)

        rf_rfecv.fit_transform(X_train, y_train)

        feature_conds_rfecv = rf_rfecv.get_support()
        transform_df = pd.DataFrame(feature_conds_rfecv)
        transform_df.to_csv(str(transform_mask) + '/RFECV_transform_SSOL6_HEA_fs_2_'
                            + str(time.strftime("%Y-%m-%d-%I-%M")) + ".csv")

        X_train = rf_rfecv.transform(X_train)

        grid_scores_df = pd.DataFrame(rf_rfecv.grid_scores_)
        grid_scores_df.to_csv(str(rf_best_params) + '/RFECV_scores_Calphad_SSOL6_HEA_fs_2_'
                              + str(time.strftime("%Y-%m-%d-%I-%M")) + '.csv')

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rf_rfecv.grid_scores_) + 1), rf_rfecv.grid_scores_)
        plt.savefig(fname=str(rf_best_params) + '/RFECV_scores_best_Calphad_SSOL6_HEA_fs_2_'
                          + str(time.strftime("%Y-%m-%d-%I-%M")) + '.png', dpi=600)

        print("Finished transforming the data; new xdf shape is: ")
        print(X_train.shape)

        Xdf = Xdf[Xdf.columns[feature_conds_rfecv]]




    if rand_search is True:

        rf = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, random_state=42, bootstrap=True)

        rscv = sklearn.model_selection.RandomizedSearchCV(rf, param_distributions={
            'n_estimators': sp_randint(5, 25),
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 3, 7, 10],
            'min_samples_split': [2, 3, 4],
            'max_features': [i for i in range(1, X_train.shape[1])]},
                                                          scoring='accuracy', cv=5, n_jobs=-1, refit=True, random_state=42,
                                                          n_iter=5000, return_train_score=True)

        print("Optimizing the Hyperparameters via Random Search CV. Please be patient.")
        yay_rand = rscv.fit(X_train, y_train)

        rand_search_df = pd.DataFrame(rscv.cv_results_)
        rand_search_df.to_csv(
            str(rf_best_params) + '/Randsearch_Calphad_HEA_fs2_post_RFECV_' + str(
                time.strftime("%Y-%m-%d-%I-%M")) + '.csv')
        rand_best_results_df = pd.DataFrame(rscv.best_params_, index=[0])
        rand_best_results_df.to_csv(str(rf_best_params) + '/Randsearch_best_Calphad_HEA_fs2_post_RFECV_' + str(
            time.strftime("%Y-%m-%d-%I-%M")) + '.csv')

        rf = sklearn.ensemble.RandomForestClassifier(**yay_rand.best_params_, random_state=42, n_jobs=-1, bootstrap=True)

        print("Optimal Hyperparameters located. Fitting model to these parameters now.")
        rf.fit(X_train, y_train)



    # ## Standard gridsearch and model fit
    # print("Building the Random Forest.")
    # rf = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, random_state=42, bootstrap=True)
    #
    # gs = sklearn.model_selection.GridSearchCV(rf, param_grid={'n_estimators': [i for i in range(11, 111, 10)],
    #                                                           'criterion': ['gini', 'entropy'],
    #                                                           'max_features': [i for i in range(1, X_train.shape[1])]},
    #                                           scoring='accuracy', cv=5, n_jobs=-1, refit=True, return_train_score=True)
    #
    # print("Optimizing the Hyperparameters. Please be patient.")
    # yay = gs.fit(X_train, y_train)
    #
    # grid_search_df = pd.DataFrame(gs.cv_results_)
    # grid_search_df.to_csv(
    #     str(rf_best_params) + '/gridsearch_Calphad_HEA_fs1_RFECV_' + str(time.strftime("%Y-%m-%d-%I-%M")) + '.csv')
    # best_results_df = pd.DataFrame(gs.best_params_, index=[0])
    # best_results_df.to_csv(str(rf_best_params) + '/gridsearch_Calphad_HEA_best_params_fs1_RFECV_' + str(
    #     time.strftime("%Y-%m-%d-%I-%M")) + '.csv')
    #
    # rf = sklearn.ensemble.RandomForestClassifier(**yay.best_params_, random_state=42, n_jobs=-1, bootstrap=True)
    #
    # print("Optimal Hyperparameters located. Fitting model to these parameters now.")
    # rf.fit(X_train, y_train)


    ## TODO: Merge grid search with RFECV
    if nested_rfecv:
        print("Determining the optimal number of features (RFECV) and hyperparameters (GS)")

        print("The original Xdf is shape: ")
        print(X_train.shape)

        param_dist = {'estimator__n_estimators': [i for i in range(11, 121, 10)],
                      'estimator__criterion': ['gini', 'entropy']}
                      # 'estimator__max_features': [i for i in range(1, 5)]}



        estimator = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, random_state=42, bootstrap=True, verbose=True, max_features='auto')

        selector = sklearn.feature_selection.RFECV(estimator=estimator, step=1, cv=5,
                                                    scoring='accuracy')

        rf_nested = sklearn.model_selection.GridSearchCV(estimator=selector, param_grid=param_dist, cv=5,
                                                            scoring='accuracy', n_jobs=-1, refit=True, return_train_score=True)


        rf_nested.fit(X_train, y_train)



        #
        features = list(Xdf.columns[rf_nested.best_estimator_.support_])
        # print(features)
        # print(len(features))


        # print(rf_nested.best_estimator_.support_)
        transform_df = pd.DataFrame(rf_nested.best_estimator_.support_)
        transform_df.to_csv(str(transform_mask) + '/Nested_RFECV_transform_SSOL6_HEA_fs_2_'
                            + str(time.strftime("%Y-%m-%d-%I-%M")) + ".csv")

        grid_scores_df = pd.DataFrame(rf_nested.cv_results_)
        grid_scores_df.to_csv(str(rf_best_params) + '/Nested_RFECV_gridscores_Calphad_SSOL6_HEA_fs_2_'
                              + str(time.strftime("%Y-%m-%d-%I-%M")) + '.csv')

        best_results_df = pd.DataFrame(rf_nested.best_params_, index=[0])
        best_results_df.to_csv(str(rf_best_params) + '/Nested_RFECV_gridscores_Calphad_HEA_best_params_fs_2_' + str(
            time.strftime("%Y-%m-%d-%I-%M")) + '.csv')


        X_train = X_train[:,transform_df[0].as_matrix()]

        Xdf = pd.DataFrame(X_train, columns=features)
        Xdf.to_excel(str(rf_best_params) + '/Nested_gs_transformed_data_' + str(time.strftime("%Y-%m-%d-%I-%M")) + '.xlsx')

        print("Finished transforming the data; new xdf shape is: ")
        print(X_train.shape)




        rf_nested_best = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, random_state=42, bootstrap=True)

        gs = sklearn.model_selection.GridSearchCV(rf_nested_best, param_grid={'n_estimators': [i for i in range(11, 121, 10)],
                                                                              'criterion': ['gini', 'entropy'],
                                                                              'max_features': [i for i in range(1, X_train.shape[1])]},
                                                  scoring='accuracy', cv=5, n_jobs=-1, refit=True, return_train_score=True)

        yay = gs.fit(X_train, y_train)

        grid_search_df = pd.DataFrame(gs.cv_results_)
        grid_search_df.to_csv(
            str(rf_best_params) + '/Nested_RFECV_gridscores_Calphad_SSOL6_HEA_fs_2_PLUS_' +
            str(time.strftime("%Y-%m-%d-%I-%M")) + '.csv')

        best_results_df = pd.DataFrame(gs.best_params_, index=[0])
        best_results_df.to_csv(str(rf_best_params) + '/Nested_RFECV_gridscores_Calphad_HEA_best_params_fs_2_PLUS_' +
            str(time.strftime("%Y-%m-%d-%I-%M")) + '.csv')

        rf_nested_best = sklearn.ensemble.RandomForestClassifier(**yay.best_params_, random_state=42,
                                                                 n_jobs=-1, bootstrap=True)

        print("Optimal Hyperparameters located. Fitting model to these parameters now.")
        rf_nested_best.fit(X_train, y_train)




        imp = rfpimp.importances(rf_nested_best, Xdf, ydf)

        viz = rfpimp.plot_importances(imp)
        viz.save(str(importance) + f'/importances_fs2_Nested_RFECV--{int(time.time())}.png')
        viz.view()

        dump(rf_nested_best, str(model_checkpoints) + '/model_checkpoint_HEA_classifier_wCALPHAD_Nested_RFECV_' + str(
            time.strftime("%Y-%m-%d-%I-%M")) + '.joblib')
        # #





    imp = rfpimp.importances(rf, Xdf, ydf)

    viz = rfpimp.plot_importances(imp)
    viz.save(str(importance) + f'/importances_fs2_Rand_search_post_RFECV--{int(time.time())}.png')
    viz.view()

    dump(rf, str(model_checkpoints) + '/model_checkpoint_HEA_classifier_wCALPHAD_fs2_Rand_search_post_RFECV_' + str(
        time.strftime("%Y-%m-%d-%I-%M")) + '.joblib')


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='path/to/data/file.xlsx')

    return parser.parse_args()


def main():
    args = parser()
    get_importance(args)


if __name__ == '__main__':
    main()
