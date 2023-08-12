AutoML and MLflow

## Client interview
Jeff HCSC looking for senior, drive initiatives, suggest approaches, architecture, perception based guy, ability to work in teams
using jira, github, team player, ownership leading initiatives
knowledge of tools i've used
mlflow, prophet, azure
things i've failed at
clean code, code review, unit tests, code quality

ability driving initiatives, how to help whole team defining standards
basic leetcode, middle element in array

teradata, minio, jenkins, mlflow, production monitoring, data lakes


## MLFlow
* MLflow integrates with all popular libraries (tensorflow, xgboost, sklearn) and can create MLFlow artifacts for your models. Data from these artifacts can be displayed in the mlflow Tracking UI that comes with the mlflow package, and can be passed as an argument to `mlflow models serve --model-uri runs:/<run-id>/model` to serve the model over http for inference
* Log parameters and outputs of runs using `mlflow.log_param("x", 1)` and `mlflow.log_metric("y", 2)` in `with mlflow.start_run()` or usually can use `mlflow.autolog()`
* Runs of a model are recorded in local files or can be sent to a remote mlflow tracking server, mlflow tracking server can be started with something like `mlflow server --backend-store-uri`
* Tracking UI lets you search and compare runs, start with `mlflow ui` or through the `mlflow server --backend-store-uri /mnt/persistent-disk --default-artifact-root s3://my-mlflow-bucket/ --host 0.0.0.0`
* Can create an MLflow project by placing a `MLproject` yml file in the root directory, this defines entry points for the project and dependencies. Can the project using `mlflow run`
* Create an MLFlow model with `MLmodel` file describing model's inputs and outputs, can save model like `mlflow.pytorch.log_model(net, "model")`
    * can run `mlflow.evaluate` which automatically saves results to local or tracking server
    * can create a docker image from a model so that it can be deployed to kubernetes
* MLFlow has a model registry for versisoning, lineage (which run produced the model), and its stage (staging, prod)
    * From the MLFlow UI you can register a model from a run details page
    * Or use `mlflow.log_model`, and retrieve it using `mlflow.load_model`
* you can look at model drift by querying runs from the MLFlow tracking server and looking for trends in the predictions, metrics, input data

## Databricks deployment, hyperparameter tuning
* MLFlow
    * You can MLFlow projects in databricks using databricks cli
    * Log runs to the `Experiments` tab in the Machine Learning profile. Set MLFlow tracking uri to the uri of the remote workspace. You can compare runs in the UI
    * register a model the same as above `mlflow.log_model`, there is a `Models` tab beneath `Experiments` tab in databricks
    * after registering model can serve it, databricks will provide endpoint to model, model can serve 3000 requests per second, serverless doesn't run in databricks cluster
* Use any hyperparameter tuning library (hyperopt, optuna, sklearn) and log the params with mlflow

## Azure ML deployment, hyperparameter tuning
* MLFlow
    * `azureml-mlflow` can be used to deploy MLFlow model to Azure ML, it will run as an Azure Container Instance or if you have AKS you can specify AKS endpoint to run there
    * Azure ML endpoint can be used as mlflow tracking server, can use any mlflow command like `log_model`, `log_param`, `log_metric` and azure ml will store it
    * Has `Experiments` view in UI to see mlflow experiments
    * You can use the mlflow package or azure ml sdk to log models, params, metrics, deploy models
* Model serving
    * create endpoint using sdk or cli
    * deploy a model to the endpoint using sdk or cli
    * need to set authentication details in deployment client (az cli or sdk) to invoke the endpoint, can't just use curl. Give consumers of endpoint a service principal
* Hyperparameter tuning is part of the azure ml sdk, can log the results using mlflow, can also uses Azure ML UI or sdk to get results of hyperparameter (sweep) job/experiment

## Model drift
* Data drift - when feature/input data to model changes from what it was trained on
* Concept drift - when definition of categories model is classifying change
* Measuring drift https://towardsdatascience.com/drift-metrics-how-to-select-the-right-metric-to-analyze-drift-24da63e497e
    * aggregate statistics such as mean, std dev. useful if the feature is normally distributed
    * density of values
    * probability density metric such as Wasserstein distance
    * Prob dist metrics like KS test, Chi-squared test (categorical features)
    * Information theory like JS distance or KL divergence
    * total variation distance
* Upstream data change - when source for data starts providing different features or stops providing some features
* Keep track of
    * distribution and schema of new training feature data, and distribution of labels in new training data
    * for prod requests keep track of distribution of requests and predictions
* You can detect data drift by periodically checking the distribution of the incoming training and request data
* You can log predictions and inference runtime in database for future evaluation
* When training the model on new data, can use MLFlow to compare model metrics to previous versions of the model


## Prophet
* https://towardsdatascience.com/time-series-analysis-with-facebook-prophet-how-it-works-and-how-to-use-it-f15ecf2c0e3a
* decomposes a time series `y(t)` into $y(t) = g(t) + s(t) + h(t) + e$ where `g` is growth, `s` is seasonality, and `h` is holidays
* growth models the overall trend of the data like linear regresssion, but pieces together several different lines at junctions called changepoints, it detects these changepoints when the data changes
* seasonality function is a fourier series, this can generate a periodic curve, more terms in series closer curve will approximate true time series
* holiday adds or subtracts a value on significant dates

## XGBoost multiple regression and time series
* see `valor/xgb_notes.md`
* for multiple regression a separate tree is trained for each `y_i`
* for time series see `league/notes.md`, can use walk forward validation where you fit model to predict next day in dataset then add that day to the training dataset, refit, predict next day, ...
* https://towardsdatascience.com/multi-step-time-series-forecasting-with-xgboost-65d6820bec39
    * to predict multiple `t+k` steps simultaneously, make your `Y` a matrix with each row `i` consisting of the `t+k` time steps out from your ith training sample and use MultiOutputRegressor


## Failures
* appending text embeddings to image embeddings in slide recommendations
* lowering inference time on product NER

## Driving initiatives
* NER model
* Slide recommendations
* Databricks workflows and performance improvements