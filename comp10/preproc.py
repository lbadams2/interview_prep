import pickle
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, MinMaxScaler, VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
from pyspark.ml import Pipeline

num_samples = 1000
num_features = 20

spark = SparkSession.builder \
    .master("local[1]") \
    .appName("test") \
    .getOrCreate()

def impute(X, numerical_cols, categorical_cols):
    imputer = Imputer(
        inputCols=numerical_cols, 
        outputCols=numerical_cols
    )
    
    X = imputer.fit(X).transform(X)
    X = X.na.fill("UNK", categorical_cols)
            
# https://stackoverflow.com/a/55258032/3614578
def clip_outliers(X, numerical_outlier_ub, numerical_outlier_lb, numerical_cols, relativeError=0.001):
    upper = numerical_outlier_ub
    lower = numerical_outlier_lb
    cols = numerical_cols
    if not isinstance(cols, (list, tuple)):
        cols = [cols]
    # Create dictionary {column-name: [lower-quantile, upper-quantile]}
    quantiles = {
        c: (F.when(F.col(c) < lower, lower)        # Below lower quantile
                .when(F.col(c) > upper, upper)   # Above upper quantile
                .otherwise(F.col(c))             # Between quantiles
                .alias(c))   
        for c, (lower, upper) in 
        # Compute array of quantiles
        zip(cols, X.stat.approxQuantile(cols, [lower, upper], relativeError))
    }

    X = X.select([quantiles.get(c, F.col(c)) for c in X.columns])
    return X

# https://stackoverflow.com/a/56953290/3614578
# https://stackoverflow.com/a/60281624/3614578
def scale_numerical_cols(X, numerical_cols):
    numerical_vars = numerical_cols
    assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in numerical_vars]
    scalers = [MinMaxScaler(inputCol=col + "_vec", outputCol=col + "_scaled") for col in numerical_vars]
    pipeline = Pipeline(stages=assemblers + scalers)
    scalerModel = pipeline.fit(X)
    scaled_x = scalerModel.transform(X)

    vec_cols = [f'{col}_vec' for col in numerical_vars]
    old_cols = numerical_vars + vec_cols
    scaled_x = scaled_x.drop(*old_cols)

    # UDF for converting column type from vector to double type
    unlist = udf(lambda x: round(float(list(x)[0]),3), FloatType())
    for col in numerical_vars:
        scaled_x = scaled_x.withColumnRenamed(f'{col}_scaled', col)
        scaled_x = scaled_x.withColumn(col, unlist(col))
    X = scaled_x

    return X


# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
def balance_classes(X):
    pass

# category_encoder or one hot vector
def enumerate_cats(X):
    pass

# use holt-winters or exponentional smoothing, remove period effected by covid and fill in that period based on time before and after
# can also use prophet to get seasonality features https://facebook.github.io/prophet/docs/quick_start.html#python-api
def normalize_covid(X):
    pass

with open('data/train.pkl', 'rb') as f:
    train_df = pickle.load(f)
train_df = train_df.sample(num_samples)
drop_cols = [f'f_{i}' for i in range(num_features, 300)]
train_df = train_df.drop(columns=drop_cols)
train_df = spark.createDataFrame(train_df)

numerical_cols = [f'f_{i}' for i in range(num_features)]
train_df = clip_outliers(train_df, .9, .1, numerical_cols)
train_df = scale_numerical_cols(train_df, numerical_cols)