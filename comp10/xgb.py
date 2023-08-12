import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost.spark import SparkXGBRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, GridSearchCV, cross_val_score
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet

num_samples = 1000
num_features = 20
cv_splits = 5

max_depth = 5 # default is 6
min_child_weight = 1 # default, min number of instances (data samples?) needed in each node
gamma = 0 # default, min loss reduction to make a further partition on a leaf node in the tree
learning_rate = .3 # default
booster = 'gbtree' # default
colsample_bytree = .75 # proportion of the features to use for a tree
tree_method = 'auto' # if gpu available try gpu_hist
objective = 'reg:squaredlogerror' # or reg:squarederror for numerical target
n_estimators = 100 # default, number of trees to create
xgb_params = {'max_depth': max_depth, 'min_child_weight': min_child_weight, 'gamma': gamma, 'learning_rate': learning_rate, 'booster': booster, \
                'colsample_bytree': colsample_bytree, 'tree_method': tree_method, 'objective': objective, 'n_estimators': n_estimators}

def bin_target_var(y):
    bins = [float('-inf'), -2, -1, 0, 1, 2, float('inf')]
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    y_cat = pd.cut(y, bins, labels=labels)
    return y_cat

def classification_model(X, y):
    y_cat = bin_target_var(y)
    le = LabelEncoder()
    y = le.fit_transform(y_cat)
    model = XGBClassifier(xgb_params)
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    model.fit(X, y)

# use model.predict_proba to view probabilities associated with each class for prediction
def regression_model(X, y):
    xgb_params['objective'] = 'reg:squarederror'
    model = XGBRegressor(**xgb_params)
    cv = KFold(n_splits=cv_splits)
    n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    model.fit(X, y)
    return model

def update_model(xgb_model):
    X, y = get_data()
    #dtrain = xgb.DMatrix(X, label=y)
    #xgb.train(xgb_params, dtrain, xgb_model=xgb_model)
    model = XGBRegressor(**xgb_params)
    model.fit(X, y, xgb_model=xgb_model)

def grid_search(X, y):
    xgb_params['objective'] = 'reg:squarederror'
    estimator = XGBRegressor()
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=xgb_params,
        scoring = 'neg_mean_absolute_error',
        n_jobs = -1,
        cv = cv_splits,
        verbose=True
    )
    grid_search.fit(X, y)

# spark df should have 2 columns - features which is a vector col, and labels for y
def distributed_train_spark(spark_df):
    spark_reg_estimator = SparkXGBRegressor(
        features_col="features",
        label_col="label",
        num_workers=2,
    )
    xgb_regressor_model = spark_reg_estimator.fit(spark_df)

# https://machinelearningmastery.com/xgboost-for-time-series-forecasting/
# use walk forward validation - fit model on first n rows and predict n+1 and calculate error
# then fit model on first n+1 rows and predict n+2, calculate and add to existing error
# can make ith row {x_i, y_i+1}, drop first and last rows
def time_series_model():
    pass

# https://facebook.github.io/prophet/docs/quick_start.html
# input is df with 2 cols, one for time (ds) and one for the metric (y)
# just call fit() on the df
def prophet_time_series(df):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=365) # this just makes a df with dates
    forecast = m.predict(future) # will have cols ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'yearly', 'monthly']
    m.plot(forecast)

def get_data():
    with open('data/train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    train_df = train_df.sample(num_samples)
    drop_cols = [f'f_{i}' for i in range(num_features, 300)]
    train_df = train_df.drop(columns=drop_cols)
    x_cols = [f'f_{i}' for i in range(num_features)]
    X = train_df[x_cols]
    y = train_df['target']
    return X, y

X, y = get_data()
#classification_model(X, y)
model = regression_model(X, y)
update_model(model)
xgb.plot_importance(model, importance_type='weight') # or cover, gain
#grid_search(X, y)