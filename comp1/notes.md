analytics and behavior change
minuteclinic, caremark prescriptions, aetna
this role affiliated with caremark
signify health physicians for house calls

drug pricing optimization models, using linear and mixed integer programming techniques, models designed to meet financial guarentees for caremark, adhering to reg and business constraints, outcomes of models are recommended drug prices

data scientist develops and deploys models

first round is 2 30 min interviews
coderpad coding interview - python and sql
second is general questions about ml, data science, stats, engineering, and optimization

next round 2 interviews
lead director of data science - general business
final is 30 minutes with exec director

4 days a month in office


## General linear models
* https://scikit-learn.org/stable/modules/linear_model.html
* General linear model has the target value as a linear combination of the features $y = b_0 + b_1 * x_1 + ... + b_n * x_n$
* OLS - solve $min_b ||Xb - y||^2$, can be solved exactly using linear algebra, features `X` should be independent for matrix `X` to have desired properties for solving. Solving uses singular value decomposition of `X`. See `league/notes.md`
* Ridge - OLS with penalty for large coeffs `b` (L2 regularization). Can make model more robust to collinearity. This cannot be solved exactly like OLS, uses iterative methods like `lbfgs`, `cholesky`, `sparse_cg`
* Lasso - similar to Ridge, adds an L1 regularization term to OLS instead of L2. Useful for sparse coefficients, more likely to set some coefficients to 0. Uses coordinate descent to solve for coefficients
* ElasticNet - uses both L1 and L2 regularization terms, combines ridge and lasso, useful for collinearity
* Generalized linear models - generalizes to accept any loss minimization, OLS is $(y - y_pred)^2$ which is assoicated with a Normal distribution. Other distributions like Bernoulli, Poisson, etc have different losses to minimize. 
    * These distribtions refer to the distribution of the target variable `y`
    * Bernoulli for binary `y` or if `y` is probabilities
    * Categorical dist generalization of Bernoulli for multi class
    * If `y` is counts or relative frequencies you might use Poisson dist
    * If `y` is skewed and positive valued Gamma or Inverse Gaussian
    * Tweedie can take on any of the above (except Bernoulli) by changing the power in the dist
    * Need to normalize feature before fitting
    * Some dists like Poisson, Gamma and Inverse Gaussian don't support negative values, but `X*b` (y_pred) can be negative, so an inverse link function `h(X*b)` is used to make it positive, h can be the exponential function
* Multiple regression - for OLS solve $min_B ||XB - Y||^2$ with B and Y matrices instead of vectors. Matrix norm (Frobenius) used

## Marketing mixed models
* https://towardsdatascience.com/market-mix-modeling-mmm-101-3d094df976f9
* https://towardsdatascience.com/introduction-to-marketing-mix-modeling-in-python-d0dd81f4e794
* https://towardsdatascience.com/an-upgraded-marketing-mix-modeling-in-python-5ebb3bddc1b6
* https://github.com/sibylhe/mmm_stan
* https://github.com/facebookexperimental/Robyn
* https://towardsdatascience.com/improving-marketing-mix-modeling-using-machine-learning-approaches-25ea4cd6994b
* can use linear regression but some variables might not have a linear impact on the dependent variable (sales)
    * depending on the nonlinear effect of the variable, can account for this by, for example, taking the log or exponential of the variable and/or adding some decay term
    * after doing this the regression equation will still be linear in the parameters $B_i$, linear regression does not have to be linear in the independent variables $x_i$
* base sales are sales without promotions, incremental sales are additional sales created by promotions
* contributions from independent variables (marketing inputs) measured by multiplying $B_i*x_i$
* $sales = B_1*x_1 + B_2*x_2 + ... + base_sales$, after regression solve this equation to get `base_sales`
* if/when coeffs from model don't give completely accurate predictions, can multiply by correction factor to get contributions 
* simple linear regression without feature engineering implies that infinite spending on advertising will increase sales infinitely
    * there are diminishing returns for advertising, model this with saturation function `1-exp(a * x_i)`
    * there is a lag effect for advertising, it doesn't effect sales instantly, carryover ad spending into future dates to account for this
    * after accounting for diminishing returns you can optimize spending for the ad channels on the trained model using `scipy.optimize.minimize` or `optuna` with some budget constraints
* features to include besides ad channel spending are price, seasonality, holidays, ...
* alternate optimization methods https://machinelearningmastery.com/optimize-regression-models/. OLS and SGD are optimization methods 
* often collinearity in features in MMM, highly correlated vars should be removed in regression
* xgboost can handle correlated vars and also non linear relationships, don't need to adjust for diminishing returns of ad channels
* normally linear regression used because parameters are directly interpretable
* cannibalization - can look at impact of sales time series of products when another goes on promo
* evaluation
    * mape - sums `(y - y_pred) / y` for each sample and divides by number of samples
    * npe - 

## Propensity matching and Control selection
* reduces bias by creating balanced groups that are equivalent on average
    * studying effects of smoking may be naturally biased because people who smoke are older and usually male (confounding factors), propensity matching can be used to create balanced groups to observe
    * a treatment is something the data has been exposed to, like smoking or maybe a promotion?
    * i think a treatment in CX could be the different promos, propensity scoring or metalearning can assign weights to the promos for their efect on the target variable sales
    * Propensity matching has to mix products on promo and not in promo in the groups
    * create counterfactuals using log reg to determine whether a sample is a member of the treatment group
    * CX used caliper matching to match members of the treatment to non members (comparison) to create heterogeneous groups
    * After creating groups check that covariates (other features) are balanced across the treratment and comparison groups
    * Propensity score is the probability of a sample being assigned to a treatment (promo)

## Mixed integer programming
* https://towardsdatascience.com/mixed-integer-linear-programming-1-bc0ef201ee87
* Solutions to these problems have a discrete solution space so optimization algorithms like SGD not suitable
* an example of a problem for MIP is maximizing profit when investments (X) have a fixed cost, can't buy fractional assets
* the features (X) do not have to be integers, they can be rational numbers (not real)
* a MIP formulation consists of constraints and an objective to optimize (for example budget and profit)
* there is a python package called `mip`
* you can add more constraints like 2 assets can't be bought together
* knapsack problem is MIP

## Time series considerations
* for cross validation need to use type compatible with time series

## How is uplift and ad channel contribution calculated for xgboost?
* model produces no interpretable params
* promo uplift and baseline calculated by modifying data with counterfactuals (what if no promo) and training on that data, see `commercial_x_2/readme.md`
* https://towardsdatascience.com/heterogeneous-treatment-effect-and-meta-learners-38fbc3ecc9d3
* uses metalearners `xlearner`, `slearner`, `tlearner` from `econml` package to infer causality for features in predicting sales, these metalearners take the trained `xgboost` model as input and assign some coefficients to the features like in regression so uplift can be calculated
    * calculates conditional average treatment effect E[Y|X,T], a coefficient for the treatment features (promos) that can be interpreted just like linear regression coefficients
    * `slearner` and `tlearner` are simpler and do not use propensity score
    * `xlearner` 
* Maximize E[Y|X,T] sales given features and treatment (promo), T is a subset of X

## Price elasticity
* Can use the coefficient of the price variable if using linear regression
* How price changes with other features (temp for example) can be found by adding an interaction term in the regression, like `price * temp`

## how is certain population affected by price change?

## Function minimization/optimization
* pyomo, `scipy.optimize.minimize`
* SGD
    * useful when number of samples and features is large
    * see `strong/notes.md`
* lbfgs
* cholesky
* sparse_cg
* coordinate descent