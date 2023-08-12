search information retrieval team
bring search business in house - data sciencce to get insight into searching
take customer query in seach bar and parse intent, inferring intent of query if not explicit
pass on to search index
responsible for autocompletion - personalize for logged in users and entity recognition
product recommendation and ranking - multimodal recommendation and identification
research training, development, prd, monitor of services

tech stack - on prem infra, kubernetes, migrate to databricks, pytorch

triton inference server

b2b searches - contextual info about them
many purchasers within a business

images to distinguish aesthetic differences between product with similar description


## intent of search query
* Use foundation language model to encode queries into matrices, a vector for each token
* Can concatenate other features onto query embedding such as season
* Create DNN (RNN or CNN) to classify these matrices into intent categories, need labeled data for this
* Can automatically create a training dataset by tracking links clicked following a query, or use word embeddings in query and match them to the embedding of the words in the category

## search index
* https://en.wikipedia.org/wiki/Search_engine_indexing
* can use an inverted index which associates pages with words
* look for important html tags like `title`
* data structure like trie that supports sequences can be used
* in addition to the tokens in the query and page, additional variables like sentiment or intent can be used to map query to page
* can take into account user behavior if logged in

## ranking pages
* https://towardsdatascience.com/learning-to-rank-a-complete-guide-to-ranking-using-machine-learning-4c9688d370d4
* take personal user information into account, location
* get the documents returned by the index and predict score for each query/doc pair
* can embed query and document into vectors and use cosine similarity
* to train an ML model for ranking
    * need training dataset consisting of a query with a list of documents, need some ground truth labels for score of each doc
    * can input single query/doc pair to network to learn the score (regression)
    * can input query with 2 docs to learn which doc is more relevant (binary classification, did model choose the most relevant)
    * can input query with list of docs 
    * sorting is non differentiable, problem when more than 1 doc input

## autocompletion
* tensorflow.js 
* there are open source js packages for this
* predict next token, append, predict next, and so on


## product recommendation and ranking, multimodal
* https://developers.google.com/machine-learning/recommendation/dnn/softmax
* https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed
* https://jorgepit-14189.medium.com/content-based-recommender-systems-in-tensorflow-and-bert-embeddings-e225fda175d1
* https://medium.com/@prateekgaurav/step-by-step-content-based-recommendation-system-823bbfd0541c
* https://arxiv.org/pdf/1409.2944.pdf
* use content based recommendation by comparing similarity of attributes of past products to recommended products
    * data for this has features for product as columns and clusters/ranks similar products to suggest a list of products given a product
* use collaborative filtering which tracks interactions such as clicks to recommend products
    * data for this has users as rows and products as columns, marks a 1 in the cell if user is interested in that product
* can cluster similar users and recommend products based on products others users liked
* looks for user-item pairs, DNN can be trained on ratings users gave to products, users can be represented by attributes
* collaborative filtering only works for users and products that have lots of data


## classic ML recommendation
* can use relational matrices, text descriptions embedded with TFIDF, can concatenate TFIDF vector with categorical features and use cosine similarity



## ner