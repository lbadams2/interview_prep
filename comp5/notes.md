## Data science questions
* k-fold cross validation data divided into k parts, loop k times and in each iteration one of the k parts used for testing and other k-1 parts used for training
* stratification used if y is categorical to ensure each fold is balanced
* For hyperparameter tuning, the cv in `GridSearchCV` means k-fold cv will run on each combination of hyperparams