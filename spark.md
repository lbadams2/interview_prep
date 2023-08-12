## CX Performance
* This https://mckinsey-hub.slack.com/archives/C03A29KTURE/p1674838940725239 can improve the seasonality pipeline performance

* The modeling performance might be improved by using SparkXGBRegressor instead of XGBRegressor https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html#sparkxgbregressor

* I think this parameter books.retail.n_partitions in books.retail and books.spine could be conflicting with spark.sql.shuffle.partitions: auto from spark.yml (auto means 200 I think). The first parameter is being used in the code to generate the books like spine_df.repartition(n_partitions) . This repartition function is apparently very time consuming. If its being done for a reason then apparently spine_df.coalesce(n_partitions) is faster for reducing the number of partitions. If its not being done for a reason one thing to try would be to remove this n_partitions param and increase spark.sql.shuffle.partitions to be 2-3 times the total cores in the cluster. Some more info here https://stackoverflow.com/a/45704560/3614578

* There's a parameter called max_train_size that you can add to the GroupTimeSeriesSplit above that will limit the amount of training data in each split. Looking at GroupTimeSeriesSplit.split it looks like the amount of training data gradually increases for each split

* It seems like at some point a grid search in the modeling pipeline should no longer be necessary. Once we've run the models on all the data, adding a smaller amount of incremental data shouldn't change the ideal hyperparameters, so the list of options in modeling/parameters_item.yml can be set to individual values

## General performance
* https://sparkbyexamples.com/spark/spark-performance-tuning/
* https://www.analyticsvidhya.com/blog/2020/11/8-must-know-spark-optimization-tips-for-data-engineering-beginners/
* Use `coalesce` to reduce partitions after filtering your data
* `mapPartitions` instead of `map`
* use `cache` on dataframe operations so they can be reused
* use `broadcast` for large read only variables, this will cache the var across all nodes
* join large to small https://stackoverflow.com/questions/44929686/join-relatively-small-table-with-large-table-in-spark-2-1

## Incremental preprocessing
* CX reruns feature engineering/preprocessing on full dataset when retraining model

## Train/test feature engineering
* https://datascience.stackexchange.com/questions/80770/i-do-feature-engineering-on-the-full-dataset-is-this-wrong
* Need to avoid data leakage between train and test sets, can't use cross row features such as mean, std, etc on whole dataset, need to split datasets first