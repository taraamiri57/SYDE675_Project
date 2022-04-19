import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import logging
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import keras.backend as K
import openpyxl
import plotly.express as px



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

# Chest Radiographs
df = pd.read_excel('Data1.xlsx', 'Sheet1')
df = df[['ReportTextText', 'Result_Infiltraat']]

# Fracture Radiographs
# df = pd.read_excel('Data2.xlsx', 'Gegevens')
# df = df[['ReportText', 'Label']]
print(df)


#prepare train-test-sets
#df pos neg split
# Chest Radiographs
df_pos = df.query('Result_Infiltraat == 1')
df_neg = df.query('Result_Infiltraat == 0')

# Fracture Radiographs
# df_pos = df.query('Label == 1')
# df_neg = df.query('Label == 0')


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
# Chest Radiographs
df['WordCount'] = df['ReportTextText'].str.split().str.len()
df_WORDS = df['WordCount'].value_counts()
print("df_WORDS:", df_WORDS)


#Counting the number of word of each text
# Fracture Radiographs
# df['WordCount'] = df['ReportText'].str.split().str.len()
# df_WORDS = df['WordCount'].value_counts()
# print("df_WORDS:", df_WORDS)

#Plot the histogram of word count to see how word count of each text affects the results of
# Fracture Radiographs
# df.sort_values(by=['Label'], inplace=True, ascending=False)
# fig = px.histogram(df, x="WordCount", color="Label")

# Chest Radiographs
df.sort_values(by=['Result_Infiltraat'], inplace=True, ascending=False)
fig = px.histogram(df, x="WordCount", color="Result_Infiltraat")
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
  tf.keras.layers.GlobalAveragePooling1D(name='Pooling-2'),
  tf.keras.layers.Dense(24, activation='relu', name='Dense-1'),  # 24
  tf.keras.layers.Dense(1, activation='sigmoid', name='Dense-2')
 ])


 model_dense.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 model_dense.summary()
 model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 model_lstm.summary()
 model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 model_cnn.summary()


 return (model_dense, model_lstm, model_cnn)




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
 for item in range(len(datastore_train)):
  temp_train = datastore_train.iloc[item]

  # Fracture Radiographs
  # training_sentences_fixed.append(temp_train['ReportText'])
  # training_labels_fixed.append(temp_train['Label'])

  # Chest Radiographs
  training_sentences_fixed.append(temp_train['ReportTextText'])
  training_labels_fixed.append(temp_train['Result_Infiltraat'])

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

  # Fracture Radiographs
  # testing_sentences_fixed.append(temp_test['ReportText'])
  # testing_labels_fixed.append(temp_test['Label'])

  # Chest Radiographs
  testing_sentences_fixed.append(temp_test['ReportTextText'])
  testing_labels_fixed.append(temp_test['Result_Infiltraat'])

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


Models = [model_dense, model_lstm, model_cnn]
Model_names = ['Dense', 'LSTM', 'CNN']
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
