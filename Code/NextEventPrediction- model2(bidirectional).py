# -*- coding: utf-8 -*-
"""Model2(Bidirectional)

This script trains a one-layer Bi-LSTM model on one of the data files in the data folder of this repository.
To change the input file to another one from the data folder, remember to indicate its name in the script.
 
It is best to run the scripts on GPU (especially when using big logs), as recurrent networks are quite computationally intensive. 
The script uses the trained LSTM model and predicts the next event for a trace, also the continuation of a trace, i.e. its suffix, 
until its completion. 

It evaluates the performance of the next event prediction, and returns the average accuracy and loss.

Author: Khadijah M. Hanga
"""

#Import libraries

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
import pydotplus as pydot
from keras.layers import Dropout

###reading the Log
df = pd.read_csv ('Helpdesk Log.csv') #Indicate file name here and ensure correct file location.
print(df.head())

###reading the VINST/BPM2013 Log
# df = pd.read_csv ('BPIC2013I.csv', encoding='latin-1', sep=';')

# # Get ndArray of all column names 
# columnsNamesArr = df.columns.values
# # Modify the Column Names, essential for data prep
# columnsNamesArr[3] = 'event'
# columnsNamesArr[0] = 'case'
# print(df.head())

###Data preparation

_ncols = ('X', 'Y')
_activeCase = "NULL"
maindfObj = pd.DataFrame([], columns=_ncols)

_tempxy = []

def create_input_output(xy, case_id):
    global maindfObj
    # Define Empty List
    values = []
    xList = [];
    #print(xy)
    values.append(("NULL", xy[0]))    
    i = 0
    while i < len(xy):
        try:
            xList = xy[0: i+1]
            xList.insert(0, "NULL")
            values.append((xList, xy[i + 1]))
        except:
            xList = xy[0: i+1]
            xList.insert(0, "NULL")
            values.append((xList, "END"))
        i = i + 1

    #print("values: ",values)

    subdfObj = pd.DataFrame(values, columns=_ncols)
    maindfObj = maindfObj.append(subdfObj)


for index, row in df.iterrows():      
      if 'case' in row and (row['case'] == _activeCase or _activeCase == "NULL"):
          concatenatedString = row['event']
          _tempxy.append(concatenatedString)
          _activeCase = row['case']
          
      else:
        create_input_output(_tempxy, _activeCase)
        _activeCase = row['case']
        _tempxy.clear()
        concatenatedString = row['event']
        _tempxy.append(concatenatedString)

Prep_data = maindfObj

#Convert activities/events to tokens.
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(Prep_data['X'])
X = tokenizer.texts_to_sequences(Prep_data['X'])
word_index = tokenizer.word_index
print(word_index)
print('Found %s unique tokens.' % len(word_index))

#Use padding to ensure all sequences are of same length.
X = pad_sequences(X)
print(X.shape)

#Convert categorical data into dummy or indicator variables.
Y = pd.get_dummies(Prep_data['Y'])
print(Y.shape)

#customised train/test split to ensure cases are intact

def append_to_2d(former_2d, new_2d):
  for i in range(len(new_2d)):
    former_2d.append(new_2d[i])
  return former_2d
def custom_splitV2(X, Y, test_size):
  Xtrain = []
  Ytrain = []
  Xtest = []
  Ytest = []
  size = X.shape  
  import random  
  #print(t)
  startList = [];
  endList = [];
  for i in range(size[0]):
    #if (i in t):
    consid = X[i]
    if consid[len(consid) - 2] == 0:
      startList.append(i)
      if(i > 0):
        endList.append(i-1)
        #print(i-1)
  endList.append(size[0]-1) #Tail End of the Array is the last element of endList 

  num_test = int(round(len(startList)*test_size))  
  num_train = len(startList) - num_test
  
  t = random.sample(startList, num_test)
  counter = 0
  for i in startList:
    Xcase = np.array(X[i:endList[counter]+1])
    Ycase = np.array(Y[i:endList[counter]+1])
    if (i in t):
      Xtest = append_to_2d(Xtest, Xcase)
      Ytest = append_to_2d(Ytest, Ycase)
      
    else:
      Xtrain = append_to_2d(Xtrain, Xcase)
      Ytrain = append_to_2d(Ytrain, Ycase)
      
    counter = counter + 1
  return np.array(Xtrain), np.array(Xtest), np.array(Ytrain), np.array(Ytest)
X_train, X_test, Y_train, Y_test = custom_splitV2(X, Y, 0.3)#split size can be changed
print("New Shape")
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

###Building Model

#required arguments for the embedding layer
MAX_NB_WORDS = 15
EMBEDDING_DIM = 32


model = Sequential()
embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1])
model.add(embedding_layer)
model.add(Bidirectional(LSTM(150, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(Y_train.shape[1], activation='softmax'))

###Compiling Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

###Training Model
print('Training...')
#history = model.fit(X, Y, validation_data=(X,Y),  epochs=50, batch_size=50, verbose=2)#without splitting to train/test
history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test),  epochs=50, batch_size=MAX_NB_WORDS, verbose=0)

### Evaluate the Model

#loss, accuracy= model.evaluate(X, Y)
loss, accuracy= model.evaluate( X_test, Y_test)
print('Loss: {:0.3f}\n  Accuracy: {:0.3f}\n '.format(loss, (accuracy*100)))

#print(embedding_layer.get_weights()[0].shape)
from matplotlib import pyplot as plt
plt.style.use('ggplot')

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

# save model to single file
model.save('Bilstm_model.h5')

###Make predictions

yhat = model.predict(X_test, verbose=0)
print(yhat)

colName = [];
for i in Y_test:
    colName.append(i)
dfObj = pd.DataFrame(list(np.round(yhat*100, decimals=0)), columns = colName)
Seq_Series=Prep_data.X.apply(pd.Series)
dfObj.reset_index(drop=True, inplace=True)
Seq_Series.reset_index(drop=True, inplace=True)
df_new = pd.concat([Seq_Series, dfObj], axis=1)
df_new.to_csv(path_or_buf= "yhat.csv", index=True)

# load model from single file
model = load_model('Bilstm_model.h5')#the model can be loaded again and used (from a different script in a different Python session)
