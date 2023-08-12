import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow_hub as hub
import tensorflow_text as text  # Imports TF ops for preprocessing.


# Define the model
BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2"
 
# Choose the preprocessing that must match the model
PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

preprocess = hub.load(PREPROCESS_MODEL)
bert = hub.load(BERT_MODEL)

def text_to_emb(input_text):
    input_text_lst = [input_text]
    inputs = preprocess(input_text_lst)
    outputs = bert(inputs)
    return np.array((outputs['pooled_output'])).reshape(-1,)


def load_data():
    # load the data, this has data about users, the ads they clicked on, and the content of the ads
    df = pd.read_csv("my_campaign.csv")

    #define the KPI
    kpi = 'clicked'

    users_features = [col for col in df if col.startswith('att_')]

    extra = ['text', 'message_id', kpi]



    # convert the df to dummies
    df = pd.concat([pd.get_dummies(df[users_features]), df[extra]], axis=1)

    # keep the unique messages that will be used for the predictions
    unique_messages = df.drop_duplicates(subset=['message_id']).sort_values(by='message_id').filter(regex='^text', axis=1)

    unique_messages_wit_ids = df.drop_duplicates(subset=['message_id','message_id']).sort_values(by='message_id').filter(regex='^text|message_id', axis=1)
    unique_messages_wit_ids.reset_index(drop=True, inplace=True)

    unique_messages_wit_ids['embeddings']  = unique_messages_wit_ids['text'].apply(lambda x:text_to_emb(x))

    # create the train and test dataset
    train=df.sample(frac=0.8,random_state=5) 
    test=df.drop(train.index)

    train.reset_index(drop=True, inplace= True)
    test.reset_index(drop=True, inplace= True)


    items_train = np.array(train.merge(unique_messages_wit_ids, how='inner', on='message_id')['embeddings'].values.tolist())
    items_test = np.array(test.merge(unique_messages_wit_ids, how='inner', on='message_id')['embeddings'].values.tolist())

    return train, test, items_train, items_test, unique_messages_wit_ids


# model has 2 DNNs, one for user and one for ads/items, 2 inputs for each, the outputs of each are dot product together
# binary classification whether user will click on recommended ad/product 
def create_model(train, items_train):
    num_user_features = train.filter(regex='^att_').shape[1]
    num_item_features =items_train.shape[1]

    # the model

    num_outputs = 32
    tf.random.set_seed(1)
    user_NN = tf.keras.models.Sequential([
        
        tf.keras.layers.Dense(128,activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dense(64,activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_outputs)

    ])

    item_NN = tf.keras.models.Sequential([
        
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_outputs)
    
    ])

    # create the user input and point to the base network
    input_user = tf.keras.layers.Input(shape=(num_user_features))
    vu = user_NN(input_user)
    vu = tf.linalg.l2_normalize(vu, axis=1)

    # create the item input and point to the base network
    input_item = tf.keras.layers.Input(shape=(num_item_features))
    vm = item_NN(input_item)
    vm = tf.linalg.l2_normalize(vm, axis=1)

    # compute the dot product of the two vectors vu and vm
    output_dot = tf.keras.layers.Dot(axes=1)([vu, vm])
    output = tf.keras.layers.Dense(1,activation='sigmoid' )(output_dot)

    # specify the inputs and output of the model
    model = Model([input_user, input_item], output)

    return model


def train_model(model, train, items_train):
    tf.random.set_seed(1)
    cost_fn = tf.keras.losses.BinaryCrossentropy()
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt,
                loss=cost_fn)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    kpi = 'clicked'
    tf.random.set_seed(1)
    model.fit([train.filter(regex='^att_').values, items_train], train[kpi].values, epochs=20,  
            batch_size=16, validation_split=0.1, callbacks=[callback] )
    
    return model
    

def predict(model, test, unique_messages_wit_ids):
    # keep the unique messages and their corresponding embeddings
    sorted_msg_ids = sorted(unique_messages_wit_ids['message_id'].values)
    unique_messages_vectors = np.array(unique_messages_wit_ids['embeddings'].values.tolist())


    preds = []
    for i in range(test.shape[0]):
        temp_pred = model.predict([np.tile(test.filter(regex='^att_').values[i], (unique_messages_vectors.shape[0],1)), unique_messages_vectors]).argmax()
        preds.append(sorted_msg_ids[temp_pred])

    return preds