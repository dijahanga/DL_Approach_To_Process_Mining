Copyright (c) 2020 Khadijah M Hanga

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# -*- coding: utf-8 -*-
"""QualityMetrics.py

This script takes as input the next activity predictions made by the lstm model
and the event log to compute the fitness, precision and fscore of the lstm model.

Author: Khadijah M. Hanga
"""

import csv
from google.colab import files
uploaded = files.upload()

import pandas as pd
import graphviz
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer

#uses the generated predictions
#read from file
predicted_dataset = pd.read_csv('predictions.csv', na_filter= False)

#Drop first column. It's irrelevant
headers = list(predicted_dataset.columns.values)
predicted_dataset = predicted_dataset.drop([headers[0]], axis=1)
print(predicted_dataset.head())

from graphviz import Digraph
import copy

def grabEventsFromHeader(header):
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
def rowIsFirst(row, activities, headers):
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

def divideMatrix(matrix):
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

def decomposeResult(results):
  headers = list(results.columns.values)
  events = grabEventsFromHeader(headers)
  matrices = []
  matricesLeft = []
  lastIndex = -1
  totalBegin = 0
  ind = 0
  newMatrix = divideMatrix(results)
  for index, row in results.iterrows():
    ind = index
    if rowIsFirst(row, events, headers):
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

def link(matrices):
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

def linkV2(matrices):
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
      if activity == 'NULL':
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

def drawGraph(transitions):
    G = Digraph('process_model', filename='graph.gv')
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
x, y = decomposeResult(predicted_dataset)
link, sequences = linkV2(y)
print(sequences)
print(len(sequences))
drawGraph(link)

with open("graph.gv") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

originalLog = pd.read_csv('helpdesk log.csv', na_filter= False)

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

originalSequence = getEventSequence(originalResult, X, Y)
print(originalSequence)
len(originalSequence)

##this step is not necessary, it saves both sequences from log and
sequences predicted in a single file for comparison sake. 
 
dfOrgSeq = pd.DataFrame(originalSequence)
dfPredSeq = pd.DataFrame(sequences)


dfOrgSeq.reset_index(drop=True, inplace=True)
dfPredSeq.reset_index(drop=True, inplace=True)
df_both = pd.concat([dfOrgSeq,dfPredSeq], axis=1)


df_both.to_csv(path_or_buf= "dfBoth.csv", index=True)
from google.colab import files
#files.download('./dfBoth.csv')

### Calculating Fitness/Recall

def findFitness(originalSequence, sequences):
  TruePositives = 0
  Count = 0
  #truePositives = 0
  #count = 0
  searched = []
  for i in range(0,len(originalSequence)):
    found = originalSequence[i] in sequences
    alreadySearched = originalSequence[i] in searched
    searched.append(originalSequence[i])
    if alreadySearched:
      Count = Count + 1
    #else:
      #count = count + 1
    if found:
      if alreadySearched:
        TruePositives = TruePositives + 1
      else:
        #truePositives = truePositives + 1
        TruePositives = TruePositives + 1
  
  print(len(originalSequence))
  #print(Count)
  print(TruePositives)
  return (TruePositives/len(originalSequence))

print(findFitness(originalSequence, sequences))

#### Calculating Precision

def findPrecision(originalSequence, sequences):
  TruePositives = 0
  Count = 0
  searched = []
  for i in range(0,len(originalSequence)):
    found = originalSequence[i] in sequences
    alreadySearched = originalSequence[i] in searched
    searched.append(originalSequence[i])
    if alreadySearched:
      Count = Count + 1
    #else:
      #count = count + 1
    if found:
      if alreadySearched:
        TruePositives = TruePositives + 1
      else:
        #truePositives = truePositives + 1
        TruePositives = TruePositives + 1
  
  #print(Count)
  print(len(sequences))
  print(TruePositives)
  return (TruePositives/len(sequences))

print(findPrecision(originalSequence, sequences))

### Calculating F-score

def findFScore():
  a = findFitness(originalSequence, sequences)
  b = findPrecision(originalSequence, sequences)
  return (2 * (a * b)/(a + b))

print(findFScore())
