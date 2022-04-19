import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from simpletransformers.classification import ClassificationModel
import logging
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import keras.backend as K
import openpyxl

# # Import generic wrappers
# from transformers import AutoModel, AutoTokenizer
#
#
# # Define the model repo
# model_name = "wietsedv/bert-base-dutch-cased-finetuned-sentiment"
#
#
# # Download pytorch model
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#
# # Transform input tokens
# inputs = tokenizer("Hello world!", return_tensors="pt")

# # Model apply
# outputs = model(**inputs)
# Define variables
vocab_size = 2500 #1000
embedding_dim = 32 #16
max_length = 250 #150 #120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size_perc = 0.8 ##2200
num_epochs_number = 12
dataset_stepsize = 100 #250
dataset_stepsize_TEST = 40
use_small_sample_perc = 1 # < 1 to us small sample of dataset for testing purpose
path = '[path]'


df = pd.read_excel('Data1.xlsx', 'Sheet1')
df = df[['ReportTextText', 'Result_Infiltraat']]



#prepare train-test-sets
#df pos neg split
df_pos = df.query('Result_Infiltraat == 1')
df_neg = df.query('Result_Infiltraat == 0')
# shuffle
df_pos_shuf = shuffle(df_pos)
df_neg_shuf = shuffle(df_neg)

#split train test
nr_pos = len(df_pos_shuf)
nr_neg = len(df_neg_shuf)
nr_train_pos = int(training_size_perc * nr_pos )
nr_train_neg = int(training_size_perc * nr_neg )
df_pos_TRAIN = df_pos_shuf.iloc[0:nr_train_pos]
df_pos_TEST = df_pos_shuf.iloc[nr_train_pos:]
df_neg_TRAIN = df_neg_shuf.iloc[0:nr_train_neg]
df_neg_TEST = df_neg_shuf.iloc[nr_train_neg:]
df_TEST = pd.concat([df_pos_TEST, df_neg_TEST])
df_TRAIN = pd.concat([df_pos_TRAIN, df_neg_TRAIN])
#safe dataset
Filename1 = 'df_TEST_THORAX'
# df_TEST.to_excel(path+'/Jupyter_NLP_thoraxdataset/Data/'+Filename1+".xlsx")
df_TEST.to_excel(Filename1+".xlsx")

Filename10 = 'df_TRAIN_THORAX'
# df_TEST.to_excel(path+'/Jupyter_NLP_thoraxdataset/Data/'+Filename1+".xlsx")
df_TRAIN.to_excel(Filename10+".xlsx")


Filename2 = 'df_pos_TRAIN_THORAX'
df_pos_TRAIN.to_excel(Filename2+".xlsx")
Filename3= 'df_neg_TRAIN_THORAX'
df_neg_TRAIN.to_excel(Filename3+".xlsx")
# #def make_list_Pos_Neg_N(pos, neg, dataset_stepsize):
# list_Pos_N = [*range(dataset_stepsize, nr_train_pos, dataset_stepsize)]
# #list_Pos_N.append(pos) # add largest number of positive cases
# list_Neg_N = [*range(dataset_stepsize, nr_train_neg, dataset_stepsize)]
# #list_Neg_N.append(neg) # add largest number of negative cases
# #return(list_Pos_N, list_Neg_N)
# #prepare results dataframe
# Training_combinations = pd.DataFrame(columns=['Dataset_ID', 'Pos', 'Neg', 'Training_size', 'Prevalence'])
# teller=1
# for i in list_Pos_N:
#  for ii in list_Neg_N:
#   ID = teller
#   Pos = round(i ,0)
#   Neg = round(ii, 0)
#   Size = round((i + ii),0)
#   Prev = round( (i/ (i + ii)), 2)
#   Training_combinations.loc[teller] = (ID, Pos, Neg, Size, Prev)
#   teller = teller + 1
# print(Training_combinations)
# Filename4 = 'Training_combinations_THORAX_20201006'
# Training_combinations.to_excel(Filename4+".xlsx")
#append info to results


print(df)

#Counting the number of word of each text
df['WordCount'] = df['ReportTextText'].str.split().str.len()
df_WORDS = df['WordCount'].value_counts()
print("df_WORDS:", df_WORDS)

#BERT
def BERTmodel2(datastore_train, output_dir_bert):
# def BERTmodel2(datastore_train):
 logging.basicConfig(level=logging.INFO)
 transformers_logger = logging.getLogger("transformers")
 transformers_logger.setLevel(logging.WARNING)
 # Create a ClassificationModel
 model_args = {
  "num_train_epochs": 4,
  "overwrite_output_dir": True,
  "save_model_every_epoch": False}
 model_BERT = ClassificationModel('bert', 'wietsedv/bert-base-dutch-cased', args=model_args, use_cuda=False)
 # Train the model
 model_BERT.train_model(datastore_train, output_dir=output_dir_bert) #other output_dir for every iteration in the loop
 # model_BERT.train_model(datastore_train)
 return( model_BERT)

#
# def make_datastore_train(nr, Training_combinations, df_pos_TRAIN, df_neg_TRAIN):
#  pos = Training_combinations.loc[nr]['Pos']
#  neg = Training_combinations.loc[nr]['Neg']
#  temp_pos = df_pos_TRAIN.loc[0:pos]
#  temp_neg = df_neg_TRAIN.loc[0:neg]
#  datastore_train = pd.concat([temp_pos, temp_neg])
#  datastore_train = shuffle(datastore_train)
#  return(datastore_train)

def make_trainset_from_datastore_train_and_testset_from_df_TEST(df_TRAIN, df_TEST):
 # training_sentences_fixed = []
 # training_labels_fixed = []
 # #teller = 0
 # for item in range(len(datastore_train1)):
 #  #print(teller)
 #  #print('item=',item)
 #  temp_train = datastore_train.iloc[item]
 #  training_sentences_fixed.append(temp_train['ReportTextText'])
 #  #print('sentences=',sentences)
 #  training_labels_fixed.append(temp_train['Result_Infiltraat'])
  #print('labels=',labels)
  #teller = teller +1
 # tokenization: basically refers to splitting up a larger body of text into
 # smaller lines, words or even creating words for a non-English language
 # tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
 # tokenizer.fit_on_texts(training_sentences_fixed)
 # word_index = tokenizer.word_index
 # training_sequences_fixed = tokenizer.texts_to_sequences(training_sentences_fixed)
 # training_padded_fixed = pad_sequences(training_sequences_fixed, maxlen=max_length, padding=padding_type, truncating=trunc_type)

 # make train datasets with tokenized reports
 training_sentences_fixed = []
 training_labels_fixed = []
 for item in range(len(df_TRAIN)):
  temp_test = df_TRAIN.iloc[item]
  training_sentences_fixed.append(temp_test['ReportTextText'])
  training_labels_fixed.append(temp_test['Result_Infiltraat'])
  # tokenizer en word-index van trainingset
  # word_index = tokenizer.word_index van trainingset
  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
  tokenizer.fit_on_texts(training_sentences_fixed)
  training_sequences_fixed = tokenizer.texts_to_sequences(training_sentences_fixed)
  training_padded_fixed = pad_sequences(training_sequences_fixed, maxlen=max_length, padding=padding_type,
                                       truncating=trunc_type)
 Tokenizer_Ext = tokenizer


 #make test datasets with tokenized reports
 testing_sentences_fixed = []
 testing_labels_fixed = []
 for item in range(len(df_TEST)):
  temp_test = df_TEST.iloc[item]
  testing_sentences_fixed.append(temp_test['ReportTextText'])
  testing_labels_fixed.append(temp_test['Result_Infiltraat'])
  # tokenizer en word-index van trainingset
  #word_index = tokenizer.word_index van trainingset
  testing_sequences_fixed = tokenizer.texts_to_sequences(testing_sentences_fixed)
  testing_padded_fixed = pad_sequences(testing_sequences_fixed, maxlen=max_length, padding=padding_type, truncating=trunc_type)
 Tokenizer_Ext = tokenizer
 return(training_padded_fixed, training_labels_fixed, testing_padded_fixed, testing_labels_fixed, Tokenizer_Ext)
# train models(Dense, LSTM, CNN) and return histories

# predict BERT (for evaluation)
def predictBERT(df_TEST, model):
  predictions, raw_outputs = model.predict(df_TEST)
  return (predictions, raw_outputs)

def evaluate_BERT2(y_true, y_pred):
  precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
  fscore_0 = fscore[0]
  f1_score = fscore[1]
  npv = precision[0]
  ppv = precision[1]
  spec = recall[0]
  sens = recall[1]
  auc = roc_auc_score(y_true, y_pred)
  return (sens, spec, ppv, npv, auc, f1_score)

# BERT
Filename1 = 'df_TEST_THORAX'
Filename2 = 'df_pos_TRAIN_THORAX'
Filename3 = 'df_neg_TRAIN_THORAX'
# Filename4 = 'Training_combinations_THORAX'
# Training_combinations = pd.read_excel(Filename4 + ".xlsx")
df_TEST = pd.read_excel(Filename1 + ".xlsx")
df_pos_TRAIN = pd.read_excel(Filename2 + ".xlsx")
df_neg_TRAIN = pd.read_excel(Filename3 + ".xlsx")
Evaluation = pd.DataFrame(columns=['ID', 'Nr', 'Training_size', 'Prevalence', 'Model', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'AUC', 'F1_score'])
# eerste = 1  #
# laatste = 31  #
# modelname = 'BERT'
# Count = 0
# for j in range(eerste, laatste):
#  nr = j - 1
#  Count = Count + 1
#  datastore_train = make_datastore_train(nr, Training_combinations, df_pos_TRAIN, df_neg_TRAIN)
#  datastore_train = datastore_train[['ReportTextText', 'Result_Infiltraat']]
#  prev = Training_combinations.loc[nr]['Prevalence']
#  size = Training_combinations.loc[nr]['Training_size']
#  print('prev=', prev)
#  print('size=', size)
#  print('nr=', nr)
#  print("datastore:", datastore_train)

output_dir_bert = "/Users/taraamiri/PycharmProjects/Project675/BERT"
df_TEST = df_TEST[['ReportTextText', 'Result_Infiltraat']]
model_BERT = BERTmodel2(df_TRAIN, output_dir_bert)
# model_BERT = BERTmodel2(datastore_train)
uitkomst, ruwe_data = predictBERT(df_TEST['ReportTextText'], model_BERT)
y_true = df_TEST['Result_Infiltraat']
y_pred = pd.DataFrame(uitkomst)
sens, spec, ppv, npv, auc, f1_score = evaluate_BERT2(y_true, y_pred)
Evaluation = (sens, spec, ppv, npv, auc, f1_score)
now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M")
filename5 = 'Evaluation_BERT' + dt_string
print('filename5=', filename5)
Evaluation.to_excel(filename5 + '.xlsx')

