Any modeling opportunities?
Recommendation system for financial advisors? Main job for this role

smart advisor platform - generate leads for clients
access to financial planning and literacy tools

python assessment
then 2 virtual rounds (1: case study, 2: talk with director and product manager)

## AWS Model deployment
* store model in S3 and and ECR image for its runtime environment
* can choose instance type and number of instances or serverless (both support up to 60 second inference time)
    * for sustained traffic and low latency don't use serverless
* Sagemaker has python sdk to do all this
* also has batch inference jobs which will write inference results to S3, these jobs can also be used for preprocessing
* Sagemaker MLOps allows you to make workflows and automatically trigger from github webhook
* can use cloudwatch to monitor errors and latency time
* Model Monitor can detect data and model drift
    * for data drift provide baseline data and it will compare incoming data metrics (mean, stddev, etc) with baseline
    * for incremental training it can monitor model metric drift
    * for inference in production it can monitor the distribution of predictions and if those have changed from training
* Model Registry allows you to asssociate metrics with a version of the model


## Canary deployment
* Rollout new feature to subset of users
* Can do this in kubernetes using the weight annotation on the nginx ingress controller
* Basically same as A/B testing