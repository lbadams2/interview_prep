## XGBoost
* works well on structured or tabular data for classification or regression predictive modeling
* some features
    * sparse aware implementation with automatic handling of missing values
    * block structure to support parallelization of tree construction
    * further train already fitted model with new data
    * can train on a cluster of machines
* uses decision trees for the models in the ensemble
* supervised learning

## Decision trees
* leaves represent class labels and branches are conjuctions of features that led to those class labels
* each non leaf node is labeled with an input feature, the leaf nodes are values of the target feature
* seems to require binary trees
* split on an input feature x_i based on gini or information gain, essentially how well it can split the target variable's values between the left and right subtrees. If the target variable has 5 values each occurring 20 times, a good split would have a large proportion of the values on either side, wouldn't want left and right subtree to both have 10 occurrences of each value.
* if input feature x_i is not a binary feature and has for example 5 values and x_i is chosen to split, the left subtree may for example be only for 2 of those values while the right for 3 values
* hierarchy of features chosen to split in decision tree represents the importance of those features
* Disadvantages
    * information gain is biased towards features with more values
    * they tend to overfit models on training data, pruning addresses this
    * small changes in training data can result in large changes to the tree
* Number of leaf nodes will be greater than number of values target variable y can take. If y takes true/false there will be multiple paths in the tree to arrive at each of the values
* Each node has a certain proportion of training data associated with it, at every level of the tree these proportions add to 1. There shouldn't be so many leaf nodes that each node has a very small amount of the training data, this leads to overfitting
* To create a decision tree for regression (y is a numerical variable), replace information gain(roughly occurrences of each value y can take on either subtree) with standard deviation reduction. Standard deviation reduction is the decrease in stddev after a node is split. Make splits on attributes that give highest standard deviation reduction (most homogeneous branches for values of y)
* Leaf nodes in regression decision tree can take average value of y among the subset of data associated with the leaf
* Multivariate prediction - use `MultiOutputRegressor` which fits a separate tree for each element of `y`

## Gradient boosting
* Boosting is an ensemble technique where new models are added to correct the errors made by existing models
* Gradient boosting creates new models to correct the errors made by existing models, and uses gradient descent to minimize the loss when adding new models
* One of imperfect models during training F_m that predicts $y_{pred} = F_m(x)$ can be improved by adding a new estimator h_m so that 
$F_m(x) + h_m(x) = y$, where y is the actual value associated with x
* In the above h_m is fit to the residual $y - F_m(x)$. The residuals are proportional to the negative gradients of the MSE loss function which means gradient descent can be used for gradient boosting
* The final model F(x) in the  gradient boosting algorithm is a weighted sum of M $h_m$ functions $F(x) = \sum_{m=1}^M \gamma_m h_m(x) + C$
* Finding $h_m$ not computationally feasible, so to get new $F_m$ from $F_{m-1}$ take negative gradient of loss function $L(y_i, F_{m-1}(x_i))$ and add that to $F_{m-1}$. This finds a local minimum of L which will be the steepest descent
* $\gamma_m$ changes for each $F_m$ to minimize loss
* In gradient tree boosting, each $h_m(x)$ is a decision tree
* Hyperparameters include M (number of iterations or $h_m$), depth of trees, learning rate (constant factor multiplied into the $\gamma_m$'s)
* The gradients calculated are on the loss function, the values of the gradient at each $x_i$ are the residuals $r_i$. New $h_m$ is trained on the set of ${x_i, r_i}$, so output of these decision trees will be the $r_i$ not $y_i$

## Gradient Descent
* Algorithm to find local minimum of a function, at each iteration take move in the direction of the negative gradient as that will be the steepest descent down the curve of the function. After solving for the gradient at point $a_n$ and moving in the direction of steepest descent, you'll arrive at another point on curve $a_{n+1}$

## Network analysis and clustering
* Can use relationships among companies to glean information on successful investments
* Can create a Bayesian network to get cause and effect info
* Can create clusters from data and check each cluster's ROI
* investor networks https://www.joim.com/wp-content/uploads/emember/downloads/p0599.pdf
* can use investor centrality and investor community as a feature for models

## Data
* Features can be things like amount of series-a funding, conversion rate of investors, total money raised, categorical features on founders' backgrounds
* The target variable might be ROI, or a categorical variable indicating whether or not they go to IPO, get acquired, fail, or remain private. This categorical variable can also be numerical with a probability for each category
* Most companies fail so there is a class imbalance
* Can do sentiment analysis on news articles and include this as a feature
* Factor in when investment occurred for seasonality features
* Preprocessing
    * fill NANs
    * filter out low variance columns
    * enumerate categorical columns - one hot encoding or scalar based on correlation to target value
    * clip outliers
    * class imbalance - undersampling, oversampling, smote
    * normalize - not necessary for decision trees
    * how to handle major events like covid or recession
    * use pandas corr function to see correlation of features to target
* to convert categorical to numerical, divide sum of y by the count of each value category can take

## Evaluation
* F1 score for categorical
* MSE for numerical
* importance of each feature for prediction
    * xgboost has function to show this https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
    * 3 options
        * weight - number of times feature is used to split data across all trees
        * cover - above metric weighted by number of training data points that go through those splits (how early in the tree the feature was used to split)
        * gain - average training loss reduction gained when using a feature for splitting
    * can also use the shap package which can produce graphs and show dependence between features 
    * lime another explainability tool
    * metalearning econml

## Models
* time series?
* implement a deep learning model too
* distribute xgboost
* use quantitative and qualitative features
* update trained model with new data
* bayesian network/classifier
    * notears/dynotears learns DAG or structure model and conditional probabilities between each node and all others. Can then set a node with a certain value and get the probability distribution for other nodes in the graph
    * a node is an independent variable in the data
* logistic/linear regression, lasso
    * https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
    * both solve for the coefficients in a linear combination of all the features
* prophet

## TODO
* prophet
* deep learning
* past projects