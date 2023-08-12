## Kubernetes
* container orchestration across many hosts
* integrated tools for logging and monitoring
* containers in same pod can share storage volumes
* rolling updates and automatic rollbacks for deployments
* kubernetes has driver and worker nodes like spark
    * kube api server process runs on driver node
* kube-scheduler assigns nodes to newly created pods
* pod
    * set of containers with shared storage and networking namespaces
    * linux namespaces isolate processes from each so they see different kernel resources
    * cgroup limits the resource usage of a collection of processes, containers in a pod may be in same cgroup
    * if pod has multiple containers they will be on same host
    * pod meant to run single instance of an application, scaling horizontally creates multiple pods
    * hostname of pod is the name of the pod
* kubelet - component running on worker node that interacts with api server to provide info about status of pods
* controller?
    * control loops that monitor state of cluster
    * there are controllers for kubernetes resource types (deployment, job, node, etc)
    * they manage state by interacting with the cluster api server, does not run the pods or containers itself
* service?
    * since pods are ephemeral, their hostnames and ip addresses may change at any time
    * service object associated with a group of pods, selector used to define the group
    * service controller scans for pods that match its selector and adds those pods to the service endpoint slice
    * uses TCP by default
    * gets assigned a cluster IP address
    * pods can resolve service names to cluster IPs using a kube cluster dns service
* ingress?
    * entrypoint for the cluster
    * can optionally do load balancing ssl termination
    * exposes routes from outside the cluster to services within the cluster
    * ingress controller implements the ingress using the config defined in the ingress, an ingress without an ingress controller has no effect
        * ingress controllers different from other controllers in that they must be explicitly deployed 
* replicaset?
    * maintain a stable set of replica pods
    * selector field in replicaset tells replicaset how to find pods to manage, this should be a field in the pod template of the pod you want to manage
    * acquires pods using the pod's OwnerReference field
    * in practice always use deployments, should never create/update replicaset objects directly
    * replaced replication controller
* deployment?
    * provides declarative updates for pods and replicasets, typically replicasets
    * you describe a desired state in a deployment, and deployment controller changes actual state to desired state at a controlled rate
    * selector field in deployment (replicaset section) tells replicaset how to find pods to manage, this should be a field in the pod template of the pod you want to manage
* stateful set?
    * manages deployment and scaling of a set of pods, provides guarantees about ordering and uniqueness of pods
    * used instead of deployments in cases where pods need persistent identities
* nodes
    * nodes run kube-proxy which is responsible for implementing a virtual IP mechanism for services, this is used instead of DNS records to avoid complications with TTL and caching. kube-proxy uses kernel resource iptables to direct traffic, updates iptables on each node when new service is created
* network policy?
    * restricts ingress or egress to/from a pod using a selector
* configmap - define non secret environment variables for pods to use
* operator - custom extensions of kubernetes that follow the control loop pattern


## FHIR and MongoDB
* monitor frequent queries - break up large documents based on data most frequently queried
* add indices
* embedding vs reference - use references for data that is rarely used otherwise embed data to make a larger document
* sharding for horizontal scaling - do this if memory usage is high
* cache read queries
* https://stackoverflow.com/a/5373969/3614578

## MLE
* avoiding overfitting
    * regularization
    * cross validation
    * dropout
    * simplify model
* regularization - adds penalty term to loss function
    * L1 (LASSO) - penalty term is sum of magnitudes of coefficients
    * L2 (Ridge) - penalty term is sum of squared magnitudes of coefficients
* nlp tokenizer splits sentences into words model understands or UNK token
* semi-supervised - cluster unlabeled data and then somehow label it
* model drift?
    * happens when target's statistical properties (mean, variance) change over time
    * incoming data probably not IID, IID means sampled from the space of all possible samples
    * maintain a static model as a baseline to compare against models being incrementally trained
    * keep track of average predictions
* model explainability?
    * deep learning
        * modify input feature values and observe differences in predictions (LIME), can compare against average of all predictions (SHAP)
        * ELI5 is a python package that can work on sklearn, xgboost and keras for above techniques
    * linear/logistic regression or any parametric model
        * coefficients are directly interpretable
    * xgboost or decision trees
        * the earlier a feature is used to make a split the more important it is
* model observability?
    * understand model at all stages of it lifecycle from prototype to production
    * similar to model drift
    * keep track of input data distribution used in training and seen in prod
    * keep track of average predictions made during training and in prod
    * to fix problems like this update training set and retrain model
    * train several candidate models and keep track of their performance on prod data
    * prometheus, cloud native services
* AutoML refers to libraries that automate finding ML pipelines for a task
    * hyperopt
* MLFlow tracks and stores models
    * can be used for model observability
* streaming data?
    * retrain model incrementally
* SGD is algorithm to find local minimum of a function, at each iteration take move in the direction of the negative gradient as that will be the steepest descent down the curve of the function. After solving for the gradient at point $a_n$ and moving in the direction of steepest descent, you'll arrive at another point on curve $a_{n+1}$

## Data Science
* time series
    * correlation of target varaiable with target from previous time step is autocorrelation
    * time series is stationary if the joint distribution over the values of the target variable doesn't change over time
    * regression
        * for single variable time series (univariate), autoregression can be done where $y$ is regressed against own lagged values. multiple lags can be used to make it a multiple variable regression
        * for time series with independent variables (multiple regression) can use lagged values of these as well to predict current $y$. this is called vector autoregression (VAR)
        * if $y$ is non stationary, meaning it has trends, can use $\delta y$ in autoregression instead of $y$
    * deep learning
        * for feed forward window data into t time steps and make target the the t+1 time step, pass all t time steps into network and output will be single neuron predicting the t+1 time step. Windows and target will be tuples like (x[0:20], x[20]), (x[1:21], x[21]), ...
        * the difference between 0 and 1 in x[0:20] and x[1:21] is the step size
        * can use same network with single neuron predictor and window input with an LSTM
        * the windowed tuples makes the time series forecasting a supervised learning problem
    * xgboost
        * Leaf nodes in regression decision tree can take average value of y among the subset of data associated with the leaf
        * make same windowed dataset as in the deep learning model to make it supervised learning, each input X to model is a list
        * like in deep learning, can use walk forward validation where you predict the first time step of the test set, record the error, add that test value to training data, refit model, repeat and aggregate errors
    * multi step forecasting
        * direct method involves building seperate model for each $t+k$ step you want to predict
        * can recursively pass $t+1$ predictions to model $k$ times
        * can create model that outputs multiple predictions to get all $t+k$ steps at once
    * multivariate time series
    * incremental training for linear/logistic regression?
* linear regression
    * $y = Xb + e$ solve for coeffs $b$ by minimizing error $e = y - Xb$, another common notation is $Ax + e = b$ minimize $(b - Ax)$
        * X or A are your observed inputs, y or b are your observed outputs or responses from the inputs, x or b unknown params you're trying to solve for
        * minimizing $(b - Ax)$ is minimizing the loss, you want the gradient of the loss to be 0 to find a minimum or maximum
    * can be nonlinear in variables, must be linear in parameters $b$
    * linear least squares (LLS) or ordinary least squares (OLS) can be used to approximate parameters $b$        
    * OLS estimator
        * if errors assumed to be normally distributed with zero mean, OLS is equivalent maximum likelihood estimator
        * objective function to minimize is $(y - Xb)^2$, this has a unique solution provided the columns of the matrix X are independent
        * the solution is given by finding the minimum value of $b$ in the normal equation $(X^TX)b = X^Ty$ -> $b = (X^TX)^{-1}X^Ty$, there is a leap from the objective function to minimize to minimizing the normal equation https://math.mit.edu/icg/resources/teaching/18.085-spring2015/LeastSquares.pdf. Normal equation comes from multiplying out $(y - Xb)^2$, taking the gradient, and setting the gradient to 0 since it will be zero at the minimum. Taking the second derivative of the normal equation confirms it is a global minimum because it is positive semi-definite
    * for multivariate linear regression, $y$ is a matrix and MLE or Bayes used as estimator see `cvs/notes.md`
* logistic regression
    * output of model can be interpreted as probability of event taking place after normalization
    * minimize sum of cross entropy losses, or maximize the likelihood function
    * fitting data to logistic function, not line. the parameters we're estimating are in the exponent of the logistic function, so it is a nonlinear equation and numerical methods are needed to find optimal values of parameters
    * logistic function used because it only takes values that are not 0 or 1 for a very small piece of the number line, its almost always either 0 or 1
    * multi class classification - see `snorkel/notes.md`
* maximum likelihood estimator - estimates parameters of an assumed probability distribution
    * selects parameters that make observed data most probable
* for small datasets, Naive Bayes classifier works well, assumes all features X are independent
* p-value - probability of getting your predicted data making some assumptions about the distribution of the data
* evaluation metrics
    * regression - mse
    * binary classification - f1 score
    * multiclass classification - f1 score per class


## Data engineer
* data warehouses more optimized for aggregation and selecting for making calculations like spark
* data warehouse schemas
    * star schema - more efficient queries than snowflake
    * snowflake schema - less storage, easier to update, more flexible
    * fact tables hold primary keys of referenced dimension tables along with metrics over which calculations are made, they correspond to events that have taken place
    * dimension tables are like customer or product tables, they are nouns 
* spark structured streaming
    * create kafka cluster to receive messages, kafka is high throughput
    * spark streaming application subscribes to kafka to ingest new data


## GCloud Spanner
* like AWS Aurora
* to speed up joins use join directives and change order of tables in joins