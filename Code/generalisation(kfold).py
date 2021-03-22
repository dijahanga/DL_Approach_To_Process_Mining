# -*- coding: utf-8 -*-
"""Generalisation(KFold)
To compute the generalisation measure, the script divides the log into k folds (k=3), 
hold out one fold, and discover the model from k−1 folds. 

The fitness of the discovered model is measured against the holdout part, 
and the precision of the discovered model against the complete log. 
The operation is repeated for all possible holdout part, and the average of scores is calculated, 
(this will lead to a k-fold fitness and a k-fold precision measure). 
The F-Score computed from k-fold fitness and k-fold precision provides a single generalization measure.

Author:Khadijah M. Hanga
"""

import pandas as pd
import graphviz
import math
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import KFold
from keras.layers import Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
import pydotplus as pydot

from graphviz import Digraph
import copy

import csv
from google.colab import files
uploaded = files.upload()

#Load file
dataset = pd.read_csv('Helpdesk.csv')
dataset.shape

#for BPIC13

#dataset = pd.read_csv ('VINST cases closed problems.csv', encoding='latin-1', sep=';')
#dataset = pd.read_csv ('VINST_cases_incidents.csv', encoding='latin-1', sep=';')
## Get ndArray of all column names 
#columnsNamesArr = dataset.columns.values
## Modify a Column Name
#columnsNamesArr[3] = 'event'
#columnsNamesArr[0] = 'case'
#print(dataset.head())
#dataset.shape

class fold:
  def __init__(self, count, dataset):
    self.count = count
    self.dataset = dataset
  def getIndexes(self, xLabel):
    indexDic = []
    indexes = []
    ev = ''
    startIndex = 0
    endIndex = 0
    for index, row in self.dataset.iterrows():      
      if ev == '':
        ev = row[xLabel]
        startIndex = index
      elif ev != row[xLabel]:
        endIndex = index - 1
        indexDic.append([startIndex, endIndex])
        startIndex = index
        ev = row[xLabel]        
    indexDic.append([startIndex, index])
    indexes = random.sample(range(0,len(indexDic)), len(indexDic))
    return indexes, indexDic
  def getFold(self, xLabel):
    indexes, indexDic = self.getIndexes(xLabel)
    apprx = round(len(indexes)/self.count)
    subDatasets = []
    k = 0    
    for i in range(0, self.count):
      innerData = 0
      if i == self.count - 1:
        apprx = len(indexes) - (apprx * (self.count -1))
      for j in range(0, apprx):
        startIndex = indexDic[indexes[k]][0]
        endIndex = indexDic[indexes[k]][1] + 1
        if type(innerData) is int:
          innerData = dataset.iloc[startIndex:endIndex]
        else:
          innerData = innerData.append(dataset.iloc[startIndex:endIndex])
        k = k + 1
      subDatasets.append(innerData)
    return subDatasets

class prepareData:
  def __init__(self, data, label):
    self.data = data
    self.label = label
  def create_input_output(self, xy):
    # Define Empty List
    values = []
    xList = [];
    _ncols = ('X', 'Y')
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
    return pd.DataFrame(values, columns=_ncols)    
  def prepare(self, test_size = 0):
    nameLabel = self.label[0]
    valueLabel = self.label[1]
    _activeCase = "NULL"
    _tempxy = []
    _ncols = ('X', 'Y')
    maindfObj = pd.DataFrame([], columns=_ncols)
    for index, row in self.data.iterrows():
      if nameLabel in row and (row[nameLabel] == _activeCase or _activeCase == "NULL"):
        concatenatedString = row[valueLabel]
        _tempxy.append(concatenatedString)
        _activeCase = row[nameLabel]
      else:
        subObject = self.create_input_output(_tempxy)
        maindfObj = maindfObj.append(subObject)
        _activeCase = row[nameLabel]
        _tempxy.clear()
        concatenatedString = row[valueLabel]
        _tempxy.append(concatenatedString)
    self.tokenize(maindfObj)
    return self.custom_split(self.X, self.Y, test_size)
  def append_to_2d(self, former_2d, new_2d):
    for i in range(len(new_2d)):
      former_2d.append(new_2d[i])
    return former_2d
  def custom_split(self, X, Y, test_size):
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    size = X.shape  
    import random
    startList = [];
    endList = [];
    for i in range(size[0]):
      consid = X[i]
      if consid[len(consid) - 2] == 0:
        startList.append(i)
        if(i > 0):
          endList.append(i-1)
    endList.append(size[0]-1) #Tail End of the Array is the last element of endList
    num_test = int(round(len(startList)*test_size))  
    num_train = len(startList) - num_test    
    t = random.sample(startList, num_test)
    counter = 0
    for i in startList:
      Xcase = np.array(X[i:endList[counter]+1])
      Ycase = np.array(Y[i:endList[counter]+1])
      if (i in t):
        Xtest = self.append_to_2d(Xtest, Xcase)
        Ytest = self.append_to_2d(Ytest, Ycase)
      else:
        Xtrain = self.append_to_2d(Xtrain, Xcase)
        Ytrain = self.append_to_2d(Ytrain, Ycase)
      counter = counter + 1
    return np.array(Xtrain), np.array(Xtest), np.array(Ytrain), np.array(Ytest)
  def tokenize(self, data):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data['X'])
    X = tokenizer.texts_to_sequences(data['X'])
    word_index = tokenizer.word_index
    X = pad_sequences(X)
    Y = pd.get_dummies(data['Y'])
    self.X = X
    self.Y = Y
    self.tokenizer = tokenizer

class training:
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    
    
    MAX_NB_WORDS =20   #50
    EMBEDDING_DIM =32   #32
    model = Sequential()    
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    self.model = model
  def train(self):
    model = self.model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    print('Training...')
    
    
    X = self.X
    Y = self.Y
    history = model.fit(X, Y,  epochs=50, batch_size=MAX_NB_WORDS, verbose=2)
    self.model = model
    return model.evaluate(X, Y)
  def crossvalidationResult(self, tokenizer, Y):
    model = self.model
    #X_test = self.X_train     #X_train
    word_index = tokenizer.word_index
    predict_proba = model.predict(X)
    colName = [];
    for i in Y:
        colName.append(i)
    dfObj = pd.DataFrame(list(np.round(predict_proba*100, decimals=0)), columns = colName)
    X_testn=tokenizer.sequences_to_texts(X)
    mother = []
    i = 0
    while i < len(X):
      j = 0
      daugther = []
      paddings = []
      while j < len(X[i]):
        if(X[i][j] == 0):
          paddings.append('')
          f = 0
        else:
          daugther.append(list(word_index.keys())[list(word_index.values()).index(X[i][j])])    
        j = j + 1
      daugther.extend(paddings)
      daughter = [daugther]
      mother.append(tuple(daughter))
      i = i + 1
    mother_Obj = pd.DataFrame(mother, columns=['C'])
    Seq_Series= mother_Obj.C.apply(pd.Series)
    dfObj.reset_index(drop=True, inplace=True)
    Seq_Series.reset_index(drop=True, inplace=True)
    df_new = pd.concat([Seq_Series, dfObj], axis=1)
    return df_new

class helper:
  def __init__(self):
    self.i= 0
  def datasetListMergeMinus(self, dataset, subset):
    wholeData = 1
    for m in dataset:
      if not m.equals(subset):
        if type(wholeData) is int:
          wholeData = m          
        else:
          wholeData = wholeData.append(m)          
    return wholeData
  def multiDimStrToUpper(self, string):
    nstring = []
    for strns in string:
      nstring.append([x.upper() for x in strns])
    return nstring
  def multiDimStrToLower(self, string):
    nstring = []
    for strns in string:
      nstring.append([x.lower() for x in strns])
    return nstring
  def grabEventsFromHeader(self, header):
    evs = []
    c = 0
    for ev in header:
      if c > 0:
        try:
          num = int(ev)
        except:
          evs.append(ev)
      c = c + 1
    return evs
  def rowIsFirst(self, row, activities, headers):
    foundValues = [];
    for i in range(0,len(headers)):
      val = headers[i]
      if val not in activities:      
        if not row[val]:        
          return True
        else:
          if len(foundValues) > 0:
            return False;
          else:
            foundValues.append(row[val])
    return False;

  def divideMatrix(self, matrix):
    headers = list(matrix.columns.values)
    leftHeaders = []
    rightHeaders = []
    for i in range(0,len(headers)):
      ev = headers[i]
      try:
          num = int(ev)        
          leftHeaders.append(ev)        
      except:
          rightHeaders.append(ev)
    leftData = matrix[leftHeaders]
    rightData = matrix[rightHeaders]
    return [leftData, rightData]

class resultGraphing:
  def __init__(self):
    self.i= 0
  def decomposeResult(self, results):
    helper_ = helper()
    headers = list(results.columns.values)
    events = helper_.grabEventsFromHeader(headers)
    matrices = []
    matricesLeft = []
    lastIndex = -1
    totalBegin = 0
    ind = 0
    newMatrix = helper_.divideMatrix(results)
    for index, row in results.iterrows():
      ind = index
      if helper_.rowIsFirst(row, events, headers):
        totalBegin = totalBegin + 1
        if lastIndex > -1:
          if lastIndex == 0:
            sequenceList = newMatrix[1].iloc[lastIndex:index, :]
            sequenceListLeft = newMatrix[0].iloc[lastIndex:index, :]
            matrices.append(sequenceList)
            matricesLeft.append(sequenceListLeft)
          else:
            sequenceList = newMatrix[1].iloc[lastIndex+1:index, :]
            sequenceListLeft = newMatrix[0].iloc[lastIndex+1:index, :]
            matrices.append(sequenceList)
            matricesLeft.append(sequenceListLeft)
        if index < 1:
          lastIndex = index
        else:
          lastIndex = index - 1
    sequenceList = newMatrix[1].iloc[lastIndex+1:ind+1, :]
    sequenceListLeft = newMatrix[0].iloc[lastIndex+1:ind+1, :]
    matrices.append(sequenceList)
    matricesLeft.append(sequenceListLeft)
    return matrices, matricesLeft

  def link(self, matrices):
    links = []
    sequences = []
    for i in range(0,len(matrices)):
      thisMatrix = matrices[i]
      lastEvent = "Start";
      sequence = []
      for index, row in thisMatrix.iterrows():
        if lastEvent == "END":
          break
        row = pd.to_numeric(row)
        evName = row.idxmax(axis=1)
        link = lastEvent + "<-->" + evName
        sequence.append(evName)
        if link not in links:
          if lastEvent != evName:
            links.append(link)          
        lastEvent = evName
      # The Last element is End and undesirable
      sequence.pop()
      sequences.append(sequence)
    return links, sequences

  def linkV2(self, matrices):
    links = []
    sequences = []
    for i in range(0,len(matrices)):
      thisMatrix = matrices[i]
      lastEvent = "Start";
      sequence = []
      lastRow = thisMatrix.tail(1)
      for column in lastRow:
        b = lastRow[column]
        activity = b.values[0]
        if activity.upper() == 'NULL':
          continue
        elif activity == '':
          activity = 'END'
        evName = activity
        link = lastEvent + "<-->" + evName
        sequence.append(evName)
        if link not in links:
          if lastEvent != evName:
            links.append(link)          
        lastEvent = evName
        if activity == 'END':
          break
      sequence.pop()
      sequences.append(sequence)
    return links, sequences

  def drawGraph(self, transitions, counter):
    G = Digraph('process_model', filename='graph_'+str(counter)+'.gv')
    G.attr(rankdir='LR', size='7,5')
    G.attr('node', shape='doublecircle', style="filled", fillcolor="grey")
    G.node('Start')
    G.node('END')
    G.attr('node', shape='box', style="bold")
    for i in range(0,len(transitions)):
      G.attr('edge', style="bold", penwidth='3.0')
      fromto = transitions[i].split("<-->")
      G.edge(fromto[0], fromto[1])
    G.view()
    return G
  
  def getEventSequence(self, data, X_label, Y_label):
    currentX_label = ''
    sequences = []
    sequence = []
    for index, row in data.iterrows():
      if currentX_label == row[X_label]:
        sequence.append(row[Y_label])
      else:
        if len(sequence) > 1:
          sequences.append(sequence)
        sequence = []
        sequence.append(row[Y_label])
      currentX_label = row[X_label]
    sequences.append(sequence)
    return sequences

originalLog = pd.read_csv('/content/Helpdesk log.csv', na_filter= False)

###reading the VINST/BPM2013 Log

#originalLog = pd.read_csv('/content/VINST cases closed problems.csv', encoding='latin-1', sep=';', na_filter= False)

## Get ndArray of all column names 
#columnsNamesArr = originalLog.columns.values
## Modify a Column Name
#columnsNamesArr[3] = 'event'
#columnsNamesArr[0] = 'case'

X = 'case' # Use this label to identify the case column, in case it has a different name
Y = 'event' # Use this label to identify the event column, in case it has a different name
def getEventSequence(data, X_label, Y_label):
  currentX_label = ''
  sequences = []
  sequence = []
  for index, row in data.iterrows():
    if currentX_label == row[X_label]:
      sequence.append(row[Y_label])
    else:
      if len(sequence) > 1:
        sequences.append(sequence)
      sequence = []
      sequence.append(row[Y_label])
    currentX_label = row[X_label]
  sequences.append(sequence)
  return sequences

original = getEventSequence(originalLog, X, Y)

class performance:
  def __init__(self):
    self.i= 0
    
  def fitness(self, holdOut, sequences):
    TruePositives = 0
    Count = 0
    searched = []
    for i in range(0,len(holdOut)):
      found = holdOut[i] in sequences      
      alreadySearched = holdOut[i] in searched
      searched.append(holdOut[i])
      if alreadySearched:
        Count = Count + 1
      #else:
        #count = count + 1
      if found:
        if alreadySearched:
          TruePositives = TruePositives + 1
        else:
          TruePositives = TruePositives + 1

    print("The fitness of the discovered model against the holdout part")
    print(" No. of True Positive: " , TruePositives)
    print(" No. of Traces in holdout: ", len(holdOut))
    return (TruePositives/len(holdOut))

  def precision(self, original, sequences):
    TruePositives = 0
    Count = 0
    searched = []
    for i in range(0,len(original)):
      found = original[i] in sequences
      alreadySearched = original[i] in searched
      searched.append(original[i])
      if alreadySearched:
        Count = Count + 1
      if found:
        if alreadySearched:
          TruePositives = TruePositives + 1
        else:
          TruePositives = TruePositives + 1

    print("The precision of the discovered model against the complete log")
    print(" No. of True Positive: ", TruePositives)
    print("No. of Traces in the model: ", len(sequences))
    return (TruePositives/len(original))

  def findFScore(self, fitness, precision):
    a = fitness
    b = precision
    return (2 * (a * b)/(a + b))

#Fold dataset in 3. 
kFold = fold(3, dataset)
c = kFold.getFold('case')
#print(c[0])

#create empty array of results
Kprecisions = []
Kfscores = []
Kfitnesses = []
#losses = []
#accuracies = []

counter = 0
#iterate over the dataset. Each dataset will undergo its own prep and training
for k in c:
  helper_ = helper()
  trainingSet = helper_.datasetListMergeMinus(c, k)
  #First Prepare the data from raw dataset
  prepdata = prepareData(trainingSet, ['case', 'event'])
  X_train, X_test, Y_train, Y_test = prepdata.prepare(0)
  #print(len(X_train), len(Y_train), len(X_test), len(Y_test))
  
  #get built tokenizer and word_index
  tokenizer = prepdata.tokenizer
  X = prepdata.X
  Y = prepdata.Y

  #Initialize model
  trainModel = training(X_train, Y_train)
  #Get loss, accuracy
  #loss, accuracy = trainModel.train()
  #losses.append(loss)
  #accuracies.append(accuracy)

  ###Get training result dataset
  resultDataset = trainModel.crossvalidationResult(tokenizer, Y)

  #Extract links and sequences from training result
  resultGraphing_ = resultGraphing()
  x, y = resultGraphing_.decomposeResult(resultDataset)
  link, sequences = resultGraphing_.linkV2(y)
  resultGraphing_.drawGraph(link, counter)

  XLabel = 'case' # Use this label to identify the case column, in case it has a different name
  YLabel = 'event' # Use this label to identify the event column, in case it has a different name
  holdOut = resultGraphing_.getEventSequence(k, XLabel, YLabel)

  #Calculate performance
  performance_ = performance()

  #In case something messed with case. 
  holdOut = helper_.multiDimStrToLower(holdOut)
  original = helper_.multiDimStrToLower(original)
  sequences = helper_.multiDimStrToLower(sequences)
    
  #Fitness and precision for individual fold in kfold
  fitness = performance_.fitness(holdOut, sequences)
  precision = performance_.precision(original, sequences)
  
  Kfitnesses.append(fitness)
  Kprecisions.append(precision)
  counter = counter + 1

#avgLoss = sum(losses)/len(losses)
#avgAccuracy = sum(accuracies)/len(accuracies)


avgFitness = sum(Kfitnesses)/len(Kfitnesses)
avgPrecision = sum(Kprecisions)/len(Kprecisions)
fscore = performance_.findFScore(avgFitness, avgPrecision)  


#print("Average Loss: "+str(avgLoss))
#print("Average Accuracy: "+str(avgAccuracy))
print("KFold Fitness: "+str(avgFitness))
print("KFold Precision: "+str(avgPrecision))
print("Generalization Score: "+str(fscore))

