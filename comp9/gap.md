* distribution of sizes (S, M, L) while optimizing sales (optimal product mix)
    * https://support.microsoft.com/en-us/office/using-solver-to-determine-the-optimal-product-mix-c057e214-962f-4339-8207-e593e340491f
    * https://smallbusiness.chron.com/determine-optimal-product-mix-32421.html
    * make ratio of sizes a variable in model?
    * create Bayesian network to do "what if?" queries
    * linear optimization with constraints (cost, demand, time to make, profit margin, returns), `pyomo`
* min cost flow - multiple constraint optmization
* roll up/down sales predictions for product categories (rollup sales forecasting or bottom up forecasting)
    * train at different levels of the product hierarchy
* time series predict n days out
    * train a different model for different values of n
    * use walk forward validation where values of the test set are being added back to the training set, and gradually make predictions until you reach n days out
    * see `tiger/notes.md`