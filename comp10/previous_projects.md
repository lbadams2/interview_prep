## CommercialX
* input is time series data of daily sales transactions for each location/item combo, predict daily total units sold for these location/item combos for some future time period
* Uses kedro pipelines - pipelines include preprocessing, feature generation, seasonality, modeling, evaluation
* runs hyperparameter tuning on prophet and xgboost models
* commx pipeline for each category of data, run each in it own spark cluster, these can be run in parallel
* each pipeline creates some groups of item/location combos and trains prophet model on each, this can be distributed across spark cluster
* automation pipeline 
    * each commx pipeline has its own set of conf files which includes hyperparameters, paths to data
    * these can be generated with cookie cutter and uploaded to dbfs
    * the commx code packaged and versioned in whl and uploaded to dbfs
    * update code to take kedro params to run partial pipelines and find conf dir
    * databricks pipeline json generated to define clusters and how many commx pipelines run in parallel
    * preprocessing only done once

## Product matching NER
* clients want to match their products with similar products from other retailers using product descriptions scraped for ecomm websites
* existing matching done using many complex regex rules
* used these regex rules to create NER training data
* fed tags and sentences to pretrained bert tokenizer, then fed this to pretrained bert model with added dropout and dense layer for our task
* created pipeline in Azure ML for preprocessing, training, and evaluation which gave access to GPU and hyperparameter tuning
* generated tagged text which can then be used to create a vector describing each product, these vectors compared for similarity

## Slide recommendations
* training data was collection of images of slides
* Goal was, given an unseen slide s, query the database of slides and return n similar slides
* Used pretrained resnet to encode images as vectors and cosine similarity to compare slides
* Also tried OCR and bert to encode text of slides and compare those vectos
* Created pipelines in Azure ML and used hyperparameter tuning to try these different models

## Convolutional neural network
* wrote CNN from scratch and distributed training across cluster using docker compose
* each batch was split and sent to workers to calculate loss for each split
* losses sent back to driver, aggregrated and back prop run through CNN to update weights

## CausalNex
* contributed PRs to this mckinsey managed open source library for constructing DAGs from time series data
* the nodes in the DAGs are for an input feature at a specific time step, the edges can go to nodes in the same time step or different time steps
* created object model for DAGs containing these type of nodes and wrote unit tests
* optimization problem is to find weighted adjacency matrices, W and A, for the input features. Adjacency matrices satisfy equation $X = XW + YA + Z$ where W and A have acyclicity constraints. Y is the time lagged data
* Once you have these adjacency matrices you can query DAG for a given value of some input feature and view the probabilty distributions of values other features can take

## Causal impact cannibalization
* identified potential cannibal/victim pairs from similar categories based on the sales increase of item on promotion and sales decrease of items not on promotion
* used `pycausalimpact` library to determine if the time series of the victim was impacted during the promotional period