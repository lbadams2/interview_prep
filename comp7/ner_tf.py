import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import tqdm
from transformers import BertTokenizerFast, TFBertModel

def tokenize(data, tokenizer, max_len):
    input_ids = list()
    attention_mask = list()
    for i in tqdm(range(len(data))):
        encoded = tokenizer.encode_plus(data[i],
                                        add_special_tokens = True,
                                        max_length = max_len,
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
    sentence = tokenizer.decode(val_input_ids[val_input_ids > 0]) # remove padded tokens having 0 value
    print("Original Text : ",str(sentence))
    print("\n")
    true_enc_tag = enc_tag.inverse_transform(y_test)

    print("Original Tags : " ,str(true_enc_tag))
    print("\n")
    
    preds = model.predict([val_input_ids,val_attention_mask])
    pred_with_pad = np.argmax(preds, axis = -1) 
    pred_without_pad = pred_with_pad[pred_with_pad>0] # remove padded tokens having 0 value
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
    enc_tag = LabelEncoder() # should probably be passed tag variable 

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') # converts words in sentences to tokens (phonemes, partial words)
    max_len = 128

    X_train, X_test, y_train, y_test = train_test_split(sentence, tag, random_state=42, test_size=0.1)
    input_ids, attention_mask = tokenize(X_train, max_len=max_len)
    val_input_ids, val_attention_mask = tokenize(X_test, max_len=max_len)

    test_tag = list()
    for i in range(len(y_test)):
        test_tag.append(np.array(y_test[i] + [0] * (128-len(y_test[i])))) # pad labels so they are all max_len in length

    train_tag = list()
    for i in range(len(y_train)):
        train_tag.append(np.array(y_train[i] + [0] * (128-len(y_train[i]))))

    bert_model = TFBertModel.from_pretrained('bert-base-uncased') # TFBertModel allows you to use keras functions
    model = create_ner_model(bert_model, max_len)

    early_stopping = EarlyStopping(mode='min', patience=5)
    history_bert = model.fit([input_ids, attention_mask], np.array(train_tag), \
        validation_data = ([val_input_ids, val_attention_mask], np.array(test_tag)), epochs = 25, batch_size = 30*2, \
            callbacks = early_stopping,verbose = True)

    # pass a single validation input to see prediction (should use test set but didn't create separate val and test sets)
    # enc_tag never initialized, this probably doesn't work
    # y_test not padded, probably should use test_tag
    test_ner(val_input_ids[0], val_attention_mask[0], enc_tag, y_test[0], bert_model, tokenizer)