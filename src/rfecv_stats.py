import pandas as pd
import numpy as np
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

input_ = '../data/P103_HEA_RF_Standard_CALPHAD_labeled.xlsx'

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

results_df = pd.DataFrame()

for i in range(100):
    print(i)
    num_of_trees = 10
    max_feat_pct = 0.30
    print("Determining the optimal number of features and what they are for {} decision trees "
          "and {} of features at each split".format(num_of_trees, max_feat_pct))

    print("The original Xdf is shape: ")
    print(X_train.shape)

    rf_base = sklearn.ensemble.RandomForestClassifier(n_jobs=-1, random_state=int(i), bootstrap=True,
                                                      n_estimators=num_of_trees, max_features=max_feat_pct)

    rf_rfecv = sklearn.feature_selection.RFECV(estimator=rf_base, step=1, cv=5, n_jobs=-1,
                                               min_features_to_select=1,
                                               scoring='accuracy', verbose=1)

    rf_rfecv.fit_transform(X_train, y_train)


    grid_scores_df = pd.DataFrame(rf_rfecv.grid_scores_)

    results_df = pd.concat([results_df, grid_scores_df], axis=1)

    print(results_df.head(10))

results_df.to_csv('rfecv_stats_100_runs_HEA_fs2.csv')
