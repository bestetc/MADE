from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lightgbm as lgb
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import QuantileTransformer

test_size = 0.25
model_path = 'C:/Users/V/Desktop/MADE/1/'

columns = ['Feature%02d'%i for i in range(0, 30)]

X = pd.read_csv('C:/Users/V/Desktop/MADE/1/train.csv', header=None)#skipinitialspace=True
y = pd.read_csv('C:/Users/V/Desktop/MADE/1/train-target.csv', header=None, dtype='int')
target = pd.read_csv('C:/Users/V/Desktop/MADE/1/test.csv', header=None)

X.columns=columns
target.columns = columns

# qt = QuantileTransformer(n_quantiles=2000, output_distribution='unifrom')
# X['']
X['target'] = y
for index, row in X.iterrows():
    if row['target'] == 1:
        X.at[index, 'Feature16'] = row['Feature16'] - 1.2442
        X.at[index,'Feature09'] = row['Feature09'] - 2.525
        X.at[index, 'Feature15'] = row['Feature15'] / 3
X = X.drop(['target'], axis=1)

X_sc = MinMaxScaler(feature_range=(0,1)).fit(X)
X_scaled = X_sc.transform(X)
X_target = X_sc.transform(target)

X_scaled = pd.DataFrame(X_scaled)
X_scaled.columns = columns
X_target = pd.DataFrame(X_target)
X_target.columns = columns

drop_cols = [
            'Feature17', 'Feature07',
            'Feature26', 'Feature11',
            'Feature01', 'Feature03', 'Feature04', 'Feature18', 'Feature20', 'Feature24','Feature25','Feature27',
            ]
    # 'Feature07', 'Feature17', 'Feature22', 'Feature26',
    #
    #noise
    # 'Feature01', 'Feature03', 'Feature04', 'Feature18', 'Feature20', 'Feature24','Feature25','Feature27',
X_scaled = X_scaled.drop(drop_cols, axis=1)
X_target = X_target.drop(drop_cols, axis=1)

X_scaled = winsorize(X_scaled,limits=.05, axis=1)

# X_scaled['target'] = y
# plt.figure(figsize=(10,8), dpi= 80)
# fig2 = sns.pairplot(X_scaled, kind="reg",
#                     hue="target"
# )
# fig2.savefig('C:/Users/V/Desktop/MADE/1/pairplot_prefit.png')
# plt.close()
# X_scaled = X_scaled.drop(['target'], axis=1)
# X_scaled = X_scaled[['Feature02', 'Feature09','Feature16','Feature29','Feature14','Feature28','Feature19',
#                      'Feature17', 'Feature21','Feature00','Feature11','Feature05',]]
# X_target = X_target[['Feature02', 'Feature09','Feature16','Feature29', 'Feature14','Feature28','Feature19',
#                      'Feature17', 'Feature21','Feature00','Feature11','Feature05',]]

# X_scaled = X_scaled.drop(drop_cols, axis=1)
# X_target = X_target.drop(drop_cols, axis=1)

y_sc = MinMaxScaler(feature_range=(0,1)).fit(y)
y_scaled = y_sc.transform(y)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size)
y_train = np.reshape(y_train, -1)
y_test = np.reshape(y_test,-1)

print(X_scaled.shape)

params = [{'num_leaves': 400,
        'learning_rate': 0.01,
        'n_estimators': 400,
        'colsample_bytree': 0.5,
        'subsample':0.5,
        'nthread': -1,
        'random_state':42}]

#     [{
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'auc', 'binary_logloss'},
#     'class_weight': 'balanced',
#     'num_boost_round': 1300, #130
#     'learning_rate':  0.01,
#     'num_leaves': 400,
#     'n_estimators': 400,
#     'max_depth': 31,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 2,
#     'reg_alpha': 0.2,
#     'reg_lambda': 0.2,
#     'verbose': 1
# }]

model = lgb.LGBMClassifier(task='train',
                        num_boost_round = 300,
                          # boosting_type = 'gbdt',
                          # max_bin = 31,
                          # num_leaves = 200,
                          # max_depth=70,
                          # n_estimators = 100,
                        n_jobs=-1,
)

gbm = model.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='auc',
                early_stopping_rounds=100,
                verbose=True,
                )

# save model to file
gbm.booster_.save_model(model_path + 'class_v1.txt')

# predict
y_pred = gbm.predict(X_scaled, num_iteration=gbm.best_iteration_)
print('The ROC-AUC of prediction is:', roc_auc_score(y_scaled, y_pred))


print(X_target.shape)
predict = gbm.predict_proba(X_target, num_iteration=gbm.best_iteration_)
predict = pd.DataFrame(predict[:,1])
predict.to_csv('C:/Users/V/Desktop/MADE/1/predict.csv', index=False, header=False)
print(predict.shape)

# feature importances
print('Feature importances:', list(gbm.feature_importances_))
feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(12, 9))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()

# Gridsearch
# print('Start training...')
# estimator = lgb.LGBMClassifier(objective = 'binary',
#                                metric = 'auc',
# )
#
# param_grid = {
#     'learning_rate': [0.06, 0.08, 0.1, 0.15, 0.2],
#     'n_estimators': [24,27, 30,33,36],
#     'num_leaves': [48, 56, 64, 72, 80],
#     'max_depth': [18, 20, 22, 24,26,28]
#     }
#
# gbm = GridSearchCV(estimator, param_grid, return_train_score=True)
# gbm.fit(X_train, y_train, verbose=1)
#
# print('Best parameters found by grid search are:', gbm.best_params_)
# print('Score is: ', gbm.best_score_)