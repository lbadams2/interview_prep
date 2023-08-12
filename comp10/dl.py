import pickle
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import tqdm
from transformers import BertTokenizerFast, TFBertModel

num_samples = 1000
num_features = 40
num_categories = 6

epochs = 20
batch_size = 16
hidden_layer_dims = [20, 10]
learning_rate = 1e-4
optimizer = 'adam'
num_categories = 6

# https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33
def feed_forward_regression(X_train, X_test, y_train):
    model = Sequential()

    # The Input Layer :
    model.add(Dense(hidden_layer_dims[0], kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

    # The Hidden Layers :
    model.add(Dense(hidden_layer_dims[1], kernel_initializer='normal',activation='relu'))

    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # Compile the network :
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split = 0.2)
    predictions = model.predict(X_test)

def bin_target_var(y):
    bins = [float('-inf'), -2, -1, 0, 1, 2, float('inf')]
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    y_cat = pd.cut(y, bins, labels=labels)
    return y_cat

# https://www.tensorflow.org/tutorials/keras/classification#build_the_model
def feed_forward_classification(X_train, X_test, y_train, y_test):
    y_cat_train = bin_target_var(y_train)
    y_cat_test = bin_target_var(y_test)

    le = LabelEncoder()
    y_cat_train = le.fit_transform(y_cat_train)
    y_cat_test = le.fit_transform(y_cat_test)

    model = Sequential([
        Dense(hidden_layer_dims[0], input_dim = X_train.shape[1], activation='relu'),
        Dense(num_categories)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(X_train, y_cat_train, epochs=epochs)
    test_loss, test_acc = model.evaluate(X_test,  y_cat_test, verbose=2)

    probability_model = tf.keras.Sequential([model, Softmax()])
    predictions = probability_model.predict(X_test)
    # np.argmax(predictions[0])

# tf dataset has window function, create this type of dataset so that current window predicts next window
# the target row/window in one pass may be part of the training data in another pass
def time_series():
    pass

def tokenize(data, tokenizer, max_len):
    input_ids = list()
    attention_mask = list()
    for i in tqdm(range(len(data))):
        encoded = tokenizer.encode_plus(data[i],
                                        add_special_tokens = True,
                                        max_length = MAX_LEN,
                                        is_split_into_words=True,
                                        return_attention_mask=True,
                                        padding = 'max_length',
                                        truncation=True,return_tensors = 'np')
                        
        
        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])
    return np.vstack(input_ids),np.vstack(attention_mask)

# feed input to pretrained model then add dropout layer and dense layer with dimension for number of entities
def create_ner_model(bert_model, max_len):
    input_ids = tf.keras.Input(shape = (max_len,),dtype = 'int32')
    attention_masks = tf.keras.Input(shape = (max_len,),dtype = 'int32')
    bert_output = bert_model(input_ids,attention_mask = attention_masks,return_dict =True)
    embedding = tf.keras.layers.Dropout(0.3)(bert_output["last_hidden_state"])
    output = tf.keras.layers.Dense(17,activation = 'softmax')(embedding) # 17 is number of entity categories
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = [output])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def test_ner(val_input_ids, val_attention_mask, enc_tag, y_test, model, tokenizer):
    val_input = val_input_ids.reshape(1,128)
    val_attention = val_attention_mask.reshape(1,128)
    
    # Print Original Sentence
    sentence = tokenizer.decode(val_input_ids[val_input_ids > 0])
    print("Original Text : ",str(sentence))
    print("\n")
    true_enc_tag = enc_tag.inverse_transform(y_test)

    print("Original Tags : " ,str(true_enc_tag))
    print("\n")
    
    preds = model.predict([val_input_ids,val_attention_mask])
    pred_with_pad = np.argmax(preds, axis = -1) 
    pred_without_pad = pred_with_pad[pred_with_pad>0]
    pred_enc_tag = enc_tag.inverse_transform(pred_without_pad)
    print("Predicted Tags : ",pred_enc_tag)

# https://www.kaggle.com/code/ravikumarmn/ner-using-bert-tensorflow-99-35
def ner():
    dataframe = pd.read_csv("/kaggle/input/entity-annotated-corpus/ner_dataset.csv",encoding = 'ISO-8859-1',error_bad_lines = False)
    dataframe = dataframe.dropna()

    sentence = dataframe.groupby("Sentence #")["Word"].apply(list).values
    pos = dataframe.groupby(by = 'Sentence #')['POS'].apply(list).values
    tag = dataframe.groupby(by = 'Sentence #')['Tag'].apply(list).values

    enc_pos = LabelEncoder()
    enc_tag = LabelEncoder()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    max_len = 128

    X_train, X_test, y_train, y_test = train_test_split(sentence, tag, random_state=42, test_size=0.1)
    input_ids, attention_mask = tokenize(X_train, max_len=max_len)
    val_input_ids, val_attention_mask = tokenize(X_test, max_len=max_len)

    test_tag = list()
    for i in range(len(y_test)):
        test_tag.append(np.array(y_test[i] + [0] * (128-len(y_test[i]))))

    train_tag = list()
    for i in range(len(y_train)):
        train_tag.append(np.array(y_train[i] + [0] * (128-len(y_train[i]))))

    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    model = create_ner_model(bert_model, max_len)

    early_stopping = EarlyStopping(mode='min', patience=5)
    history_bert = model.fit([input_ids, attention_mask], np.array(train_tag), \
        validation_data = ([val_input_ids, val_attention_mask], np.array(test_tag)), epochs = 25, batch_size = 30*2, \
            callbacks = early_stopping,verbose = True)

    test_ner(val_input_ids[0], val_attention_mask[0], enc_tag, y_test[0], bert_model, tokenizer)

def get_data():
    with open('data/train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    train_df = train_df.sample(num_samples)
    drop_cols = [f'f_{i}' for i in range(num_features, 300)]
    train_df = train_df.drop(columns=drop_cols)
    x_cols = [f'f_{i}' for i in range(num_features)]
    X = train_df[x_cols]
    y = train_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_data()
#feed_forward_regression(X_train, X_test, y_train)
feed_forward_classification(X_train, X_test, y_train, y_test)