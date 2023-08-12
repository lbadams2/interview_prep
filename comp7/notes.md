## Logistic regression
* https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
* https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
* https://en.wikipedia.org/wiki/Multinomial_logistic_regression
* http://people.tamu.edu/~sji/classes/LR.pdf
* logistic regression is a binary classifier, in sklearn to handle multiple classes it uses one vs rest or cross entropy loss depending on whats passed to the `multi_class` arg
* it is sometimes also called a log-linear classifier
* `penalty`, `solver`, and `multi_class` args have dependencies on each other
* regularization penalizes weights from getting too large, not needed in linear regression because it has closed form solution for minimizing loss (OLS), logistic regression uses iterative numerical methods
    * L1
    * L2
    * ElasticNet - combo of L1 and L2
* the `sag` and `saga` solvers use a type of SGD
* for some solvers it is important to scale/normalize the input
* intercept is bias, logistic regression is a high bias method
* collinearity assumed to be low but variables not required to be independent of each other, highly correlated variables should be removed keeping one of them
* advantage of logistic regression compared to other models is predictions can be directly interpreted as a probability value
* multinomial or multi class logistic regression https://scikit-learn.org/stable/modules/linear_model.html#multinomial-case
    * there are a matrix of parameters to solve for, instead of a vector in log reg. Vector of params for each class. `B_{i,j}` parameter for feature `i` and class `j`
    * softmax in multi class is equivalent of logistic function in binary
    * loss in multi class can be represented by cross entropy, measures the difference between 2 prob dists, true dist and predicted dist. True dist is the one hot encoded vector for the class (all 0s and a single 1), predicted dist will not have all 0s but ideally would have a value close to 1 for the index of the class
    * One vs rest - creates a series of binary classifiers, one for each class k where y_pred = 1 if it predicts class k and 0 for any other class. During inference it passes the sample to all classifiers and picks the one with the highest output
* for classification use precision, recall, accuracy, f1 score for evaluation
* can can train incrementally using something like `SGDClassifier.partial_fit`, also called online learning

## Bias/Variance tradeoff
* https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/
* https://towardsdatascience.com/two-important-machine-learning-concepts-to-improve-every-model-62fd058916b
* low bias/high variance methods have less assumptions about form of target function - decision trees, KNN, SVM
    * these will have large changes to predictions with changes to training data
    * these are normally non linear methods
    * each time you train a model with high variance you get a slightly different result (weights/coeffs)
    * 2 common sources of variance are noise in data and randomness used by model
    * measure variance by repeated k-fold cross validation
    * decrease variance by increasing training data size, decrease model size, adding regularization and dropout, ensemble
    * high variance associated with overfitting, meaning the model is too complex
    * https://machinelearningmastery.com/how-to-reduce-model-variance/
* high bias/low variance methods have more assumptions about form of target function - linear/logistic regression
    * these will have small changes to predictions with changes to training data
    * these are normally linear methods
    * to reduce bias can add more parameters to model, reduce regularization
    * high bias associated with underfitting, meaning the model is unable to capture the true relationship between the features and target


## NER
Tags are y values or labels trying to predict, X is sentences
### TF
* `tokenize` function uses pretrainied tokenizer to return input IDs and attention mask
    * input IDs are token indices, just numerical labels for the tokens
    * attention mask is a tensor indicating which indices in each sequence (sentence converted to IDs) have been padded so the model doesn't use them

### HuggingFace/Torch
can do hyperparameter tuning with the `hyperopt` package and `Trainer.hyperparameter_search`


## Automate data labeling
https://snorkel.ai/automated-data-labeling/
https://docs.aws.amazon.com/sagemaker/latest/dg/sms-automated-labeling.html
https://towardsdatascience.com/ai-assisted-automated-machine-driven-data-labeling-approach-afde67e32c52
* Active learning - choosing to label data points that are dissimilar from one another (they are more likely to contribute novel information)
* Semi supervised - use a little bit of labeled training data to predict new labels, add them to training data

## Categorical variable encoding
2 easiest are ordinal for categories with ordering aand one hot for categories that don't have ordering
https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/
https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02

## TODO
* automate labeling
* hyperparameter tuning, grid search - see `../valor/xgb.grid_search`