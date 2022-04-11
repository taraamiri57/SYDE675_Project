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

from dbn.tensorflow import SupervisedDBNClassification

import keras.backend as K
import openpyxl





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



df = pd.read_excel('Data2.xlsx', 'Gegevens')
df = df[['ReportText', 'Label']]
print(df)


#prepare train-test-sets
#df pos neg split
df_pos = df.query('Label == 1')
df_neg = df.query('Label == 0')
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
#safe dataset
Filename1 = 'df_TEST_THORAX_20201006'
# df_TEST.to_excel(path+'/Jupyter_NLP_thoraxdataset/Data/'+Filename1+".xlsx")
df_TEST.to_excel(Filename1+".xlsx")
Filename2 = 'df_pos_TRAIN_THORAX_20201006'
df_pos_TRAIN.to_excel(Filename2+".xlsx")
Filename3= 'df_neg_TRAIN_THORAX_20201006'
df_neg_TRAIN.to_excel(Filename3+".xlsx")
#def make_list_Pos_Neg_N(pos, neg, dataset_stepsize):
list_Pos_N = [*range(dataset_stepsize, nr_train_pos, dataset_stepsize)]
#list_Pos_N.append(pos) # add largest number of positive cases
list_Neg_N = [*range(dataset_stepsize, nr_train_neg, dataset_stepsize)]
#list_Neg_N.append(neg) # add largest number of negative cases
#return(list_Pos_N, list_Neg_N)
#prepare results dataframe
Training_combinations = pd.DataFrame(columns=['Dataset_ID', 'Pos', 'Neg', 'Training_size', 'Prevalence'])
teller=1
for i in list_Pos_N:
 for ii in list_Neg_N:
  ID = teller
  Pos = round(i ,0)
  Neg = round(ii, 0)
  Size = round((i + ii),0)
  Prev = round( (i/ (i + ii)), 2)
  Training_combinations.loc[teller] = (ID, Pos, Neg, Size, Prev)
  teller = teller + 1
print(Training_combinations)
Filename4 = 'Training_combinations_THORAX_20201006'
Training_combinations.to_excel(Filename4+".xlsx")
#append info to results


print(df)

#Counting the number of word of each text
df['WordCount'] = df['ReportText'].str.split().str.len()
df_WORDS = df['WordCount'].value_counts()
print("df_WORDS:", df_WORDS)

#Plot the histogram of word count to see how word count of each text affects the results of
import plotly.express as px
df.sort_values(by=['Label'], inplace=True, ascending=False)
fig = px.histogram(df, x="WordCount", color="Label")
fig.show()

print(df_TEST)

print("list_Pos_N:", list_Pos_N, "list_Neg_N:", list_Neg_N)


def make_and_compile_models():
 # Defining the layers of the model
 # dense model:1-Embedding layer 2-Flatten layer 3- 4 fully connected layers
 model_dense = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length, name='Embedding'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu', name='Dense1'),
  # tf.keras.layers.Dense(128, activation='relu'),
  # tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(16, activation='relu', name='Dense-2'),
  tf.keras.layers.Dense(8, activation='relu', name='Dense-3'),  # 24
  tf.keras.layers.Dense(1, activation='sigmoid', name='Dense-4')
 ])
 # LSTM model: 1-Embedding layer 2-Two bidirectional layers 3-Two fully connected layers
 model_lstm = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length, name='Embedding'),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True), name='LSTM-1'),  # 32
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32), name='LSTM-2'),
  tf.keras.layers.Dense(24, activation='relu', name='Dense-1'),  # 24
  tf.keras.layers.Dense(1, activation='sigmoid', name='Dense-2')
 ])
 # CNN model: 1-Embedding layer 2-A convolutional layer 3-An average pooling layer 4-A convoloutional laeyr
 # 4-Global average pooling 5-Two fully connected layers
 model_cnn = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length, name='Embedding'),
  tf.keras.layers.Conv1D(64, 5, activation='relu', name='Conv-1D-1'),  # 32
  tf.keras.layers.AveragePooling1D(name='Pooling-1'),
  tf.keras.layers.Conv1D(64, 5, activation='relu', name='Conv-1D-2'),  # 32
  # tf.keras.layers.AveragePooling1D(),
  # tf.keras.layers.Conv1D(32, 5, activation='relu'), #32
  # tf.keras.layers.AveragePooling1D(),
  # tf.keras.layers.Conv1D(32, 5, activation='relu'), #32
  tf.keras.layers.GlobalAveragePooling1D(name='Pooling-2'),
  tf.keras.layers.Dense(24, activation='relu', name='Dense-1'),  # 24
  # tf.keras.layers.Dropout(0.2),
  # tf.keras.layers.Dense(12, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid', name='Dense-2')
 ])

 # DBN model
 # model_DBN = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
 #                                          learning_rate_rbm=0.05,
 #                                          learning_rate=0.1,
 #                                          n_epochs_rbm=10,
 #                                          n_iter_backprop=100,
 #                                          batch_size=32,
 #                                          activation_function='relu',
 #                                          dropout_p=0.2)


 model_dense.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 model_dense.summary()
 model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 model_lstm.summary()
 model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 model_cnn.summary()


 return (model_dense, model_lstm, model_cnn)

model_DBN = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                          learning_rate_rbm=0.05,
                                          learning_rate=0.1,
                                          n_epochs_rbm=10,
                                          n_iter_backprop=100,
                                          batch_size=32,
                                          activation_function='relu',
                                          dropout_p=0.2)


#BERT
def BERTmodel2(datastore_train, output_dir_bert):
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
 return( model_BERT)



def make_datastore_train(nr, Training_combinations, df_pos_TRAIN, df_neg_TRAIN):
 pos = Training_combinations.loc[nr]['Pos']
 neg = Training_combinations.loc[nr]['Neg']
 temp_pos = df_pos_TRAIN.loc[0:pos]
 temp_neg = df_neg_TRAIN.loc[0:neg]
 datastore_train = pd.concat([temp_pos, temp_neg])
 datastore_train = shuffle(datastore_train)
 return(datastore_train)
def make_trainset_from_datastore_train_and_testset_from_df_TEST(datastore_train, df_TEST):
 training_sentences_fixed = []
 training_labels_fixed = []
 #teller = 0
 for item in range(len(datastore_train)):
  #print(teller)
  #print('item=',item)
  temp_train = datastore_train.iloc[item]
  training_sentences_fixed.append(temp_train['ReportText'])
  #print('sentences=',sentences)
  training_labels_fixed.append(temp_train['Label'])
  #print('labels=',labels)
  #teller = teller +1
 # tokenization: basically refers to splitting up a larger body of text into
 # smaller lines, words or even creating words for a non-English language
 tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
 tokenizer.fit_on_texts(training_sentences_fixed)
 word_index = tokenizer.word_index
 training_sequences_fixed = tokenizer.texts_to_sequences(training_sentences_fixed)
 training_padded_fixed = pad_sequences(training_sequences_fixed, maxlen=max_length, padding=padding_type, truncating=trunc_type)
 #make test datasets with tokenized reports
 testing_sentences_fixed = []
 testing_labels_fixed = []
 for item in range(len(df_TEST)):
  temp_test = df_TEST.iloc[item]
  testing_sentences_fixed.append(temp_test['ReportText'])
  testing_labels_fixed.append(temp_test['Label'])
  # tokenizer en word-index van trainingset
  #word_index = tokenizer.word_index van trainingset
  testing_sequences_fixed = tokenizer.texts_to_sequences(testing_sentences_fixed)
  testing_padded_fixed = pad_sequences(testing_sequences_fixed, maxlen=max_length, padding=padding_type, truncating=trunc_type)
 Tokenizer_Ext = tokenizer
 return(training_padded_fixed, training_labels_fixed, testing_padded_fixed, testing_labels_fixed, Tokenizer_Ext)
# train models(Dense, LSTM, CNN) and return histories
def train_models(training_padded, training_labels, testing_padded, testing_labels, model_dense, model_lstm, model_cnn):
 num_epochs = num_epochs_number # 50
 training_padded = np.array(training_padded)
 training_labels = np.array(training_labels)
 testing_padded = np.array(testing_padded)
 testing_labels = np.array(testing_labels)
 history1 = model_dense.fit(training_padded, training_labels, epochs=num_epochs, verbose=2, use_multiprocessing = False)
 history2 = model_lstm.fit(training_padded, training_labels, epochs=num_epochs, verbose=2, use_multiprocessing = False)
 history3 = model_cnn.fit(training_padded, training_labels, epochs=num_epochs, verbose=2, use_multiprocessing = False)

 return(history1, history2, history3)
#evaluation
def eval_model(model_nr, testing_padded_fixed, testing_labels_fixed):
 y_pred1 = model_nr.predict(testing_padded_fixed)
 y_true = testing_labels_fixed
 y_pred1_rounded = np.around(y_pred1) #convert prediction to 0/1 labels
 precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred1_rounded)
 fscore_0 = fscore[0]
 f1_score = fscore[1]
 npv = precision[0]
 ppv = precision[1]
 spec = recall[0]
 sens = recall[1]
 auc = roc_auc_score(y_true, y_pred1_rounded)
 return(sens, spec, ppv, npv, auc, f1_score)
#evaluate BERT
def evaluate_BERT(result_BERT, Count, DatasetN, PrevalenceN, Training_sizeN, Testing_sizeN ):
 tp = (result_BERT['tp'])
 tn = (result_BERT['tn'])
 fp = (result_BERT['fp'])
 fn = (result_BERT['fn'])
 Evaluation_BERT = pd.DataFrame(columns=['ID','Dataset', 'Prevalence', 'Training_size', 'Testing_size', 'Model', 'AUC', 'Recall_0', 'Recall_1', 'Precision_0', 'Precision_1', 'Fscore_0', 'Fscore_1', 'Balanced_accuracy' ])
 Dataset = DatasetN
 balanced_accuracy_BERT = (1 / 2) * ((tp / (tp + fn)) + (tn / (tn + fp)))
 precision_BERT = tp / (tp + fp)
 recall_BERT = tp / (tp + fn)
 fscore_BERT = 2 * ((precision_BERT * recall_BERT) / (precision_BERT + recall_BERT))
 # Evaluation_BERT is pd.dataframe that will be updated from this function (without input/export of this dataframe)
 Evaluation_BERT.loc[Count] = (Count, DatasetN, PrevalenceN, Training_sizeN, Testing_sizeN, 'BERT', 'auc', recall_BERT, recall_BERT, precision_BERT, precision_BERT, fscore_BERT, fscore_BERT, balanced_accuracy_BERT)
 # let op: recall, precision en fscore niet apart voor 0 en 1.
 return ()  # dit was het

 # return(sens, spec, ppv, npv, auc, f1_score) #dit moet het worden
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

# overview of models
model_dense_graph, model_lstm_graph, model_cnn_graph= make_and_compile_models()


#Dense, LSTM, CNN
Filename1 = 'df_TEST_THORAX_20201006'
Filename2 = 'df_pos_TRAIN_THORAX_20201006'
Filename3= 'df_neg_TRAIN_THORAX_20201006'
Filename4 = 'Training_combinations_THORAX_20201006'
Training_combinations = pd.read_excel(Filename4+".xlsx")
df_TEST = pd.read_excel(Filename1+".xlsx")
df_pos_TRAIN = pd.read_excel(Filename2+".xlsx")
df_neg_TRAIN = pd.read_excel(Filename3+".xlsx")
Evaluation = pd.DataFrame(columns=['ID','Nr', 'Training_size', 'Prevalence', 'Model', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'AUC', 'F1_score'])
eerste = 28 #
laatste = 29 # 1 extra dan einde
histories=pd.DataFrame()
Count = 0
for j in range(eerste, laatste):
 nr = j - 1

datastore_train = make_datastore_train(nr, Training_combinations, df_pos_TRAIN, df_neg_TRAIN)
training_padded_fixed, training_labels_fixed, testing_padded_fixed, testing_labels_fixed, Tokenizer_Ext =make_trainset_from_datastore_train_and_testset_from_df_TEST(datastore_train, df_TEST)
model_dense, model_lstm, model_cnn= make_and_compile_models()
history1, history2, history3= train_models(training_padded_fixed, training_labels_fixed, testing_padded_fixed, testing_labels_fixed, model_dense, model_lstm, model_cnn)
history4=model_DBN.fit(training_padded_fixed,training_labels_fixed)

Models = [model_dense, model_lstm, model_cnn]
Model_names = ['Dense', 'LSTM', 'CNN', 'DBN']
prev = Training_combinations.loc[nr]['Prevalence']
size = Training_combinations.loc[nr]['Training_size']
print('prev=', prev)
print('size=', size)
print('nr=', nr)
for iii in range(len(Models)): #loop over model evaluation with prediction
 Count = Count+1
 print(Count)
 model = Models[iii]
 modelname = Model_names[iii]
 print("model name:", modelname)
 sens, spec, ppv, npv, auc, f1_score = eval_model(model, testing_padded_fixed, testing_labels_fixed)
 Evaluation.loc[Count] = (Count, j, size, prev, modelname, sens, spec, ppv, npv, auc, f1_score )
 now = datetime.now()
 dt_string = now.strftime("%Y%m%d_%H%M")
 filename5 = 'Evaluation_'+dt_string
 print('filename5=', filename5)
 Evaluation.to_excel(filename5 +'.xlsx')
 hist1 = pd.DataFrame(history1.history)
 hist1['model']='Dense'
 hist1['size']=size
 hist1['prev']=prev
 hist2 = pd.DataFrame(history1.history)
 hist2['model']='LSTM'
 hist2['size']=size
 hist2['prev']=prev
 hist3 = pd.DataFrame(history1.history)
 hist3['model']='CNN'
 hist3['size']=size
 hist3['prev']=prev


 histories = pd.concat([histories, hist1, hist2, hist3])
 histories.to_excel('histories'+filename5+'.xlsx')

# BERT
Filename1 = 'df_TEST_THORAX_20201006'
Filename2 = 'df_pos_TRAIN_THORAX_20201006'
Filename3 = 'df_neg_TRAIN_THORAX_20201006'
Filename4 = 'Training_combinations_THORAX_20201006'
Training_combinations = pd.read_excel(Filename4 + ".xlsx")
df_TEST = pd.read_excel(Filename1 + ".xlsx")
df_pos_TRAIN = pd.read_excel(Filename2 + ".xlsx")
df_neg_TRAIN = pd.read_excel(Filename3 + ".xlsx")
Evaluation1 = pd.DataFrame(columns=['ID', 'Nr', 'Training_size', 'Prevalence', 'Model', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'AUC', 'F1_score'])
eerste1 = 1  #
laatste1 = 31  #
modelname1 = 'DBN'
Count1 = 0
for j1 in range(eerste1, laatste1):
 nr1 = j1 - 1
 Count1 = Count1 + 1
 datastore_train = make_datastore_train(nr1, Training_combinations, df_pos_TRAIN, df_neg_TRAIN)
 datastore_train = datastore_train[['ReportTextText', 'Result_Infiltraat']]
 prev1 = Training_combinations.loc[nr]['Prevalence']
 size1 = Training_combinations.loc[nr]['Training_size']
 print('prev=', prev1)
 print('size=', size1)
 print('nr=', nr1)
 # output_dir_bert = "/Users/taraamiri/PycharmProjects/Project675/"
 df_TEST1 = df_TEST[['ReportTextText', 'Result_Infiltraat']]
 # model_BERT = BERTmodel2(datastore_train, output_dir_bert)
 uitkomst, ruwe_data = predictBERT(df_TEST['ReportTextText'], history4)
 y_true = df_TEST['Result_Infiltraat']
 y_pred = pd.DataFrame(uitkomst)
 sens1, spec1, ppv1, npv1, auc1, f1_score1 = evaluate_BERT2(y_true, y_pred)
 Evaluation1.loc[Count] = (Count1, j1, size1, prev1, modelname1, sens1, spec1, ppv1, npv1, auc1, f1_score1)
 now1 = datetime.now()
 dt_string1 = now1.strftime("%Y%m%d_%H%M")
 filename51 = 'Evaluation_BERT' + dt_string1
 print('filename51=', filename51)
 Evaluation1.to_excel(filename51 + '.xlsx')