# LSTM_Approach_To_ProcessMining

Following is the supplementary material for the article “A Graph-Based Approach to Interpreting Recurrent Neural Networks in Process Mining” by K. M. Hanga, Y. Kovalchuk, and M. M. Gaber.
The code provided in this repository can be used to perform the following tasks:
•	Prediction of the next activity to be executed in a process instance.

•	Prediction of the continuation of a process instance (i.e. its suffix).

•	Generating a process model graph explaining the decision-making of the LSTM model when predicting process event sequences. 

•	Computing the similarity between the graphs to validate the generalising ability of the model.

•	Perform some process mining tasks (like)

The scripts train a Long Short Term Memory (LSTM)-based predictive model using historical data (i.e. completed process instances). The models are evaluated on test datasets.
Requirements: 
Python 3. Additionally, the following Python libraries are required to run the code: Keras (as a backend, TensorFlow, or Theano is needed), unicodecsv, numpy, sklearn, matplotlib, pydot, graphviz, h5py. (Latest versions might be mandatory).

USAGE: 
Data format
The tool assumes the input is a complete log of all traces in the CSV format which has a case ID column, and an event column containing activity names or ID. The input log is in some cases split into 70% (training set) and 30% (test set). Sample datasets used in the paper are provided in the data folder.
Model training and Evaluation: 

This script trains a one-layer LSTM model with an embedding layer on one of the data files in the data folder of this repository (the helpdesk event log is the default). To change the input file to another one from the data folder, remember to indicate its name in the script. It is best to run the scripts on GPU (especially when using big logs), as recurrent networks are quite computationally intensive. The script uses the trained LSTM model and predicts the next event for a trace. It also predicts the continuation of a trace, i.e. its suffix, until its completion. It evaluates the performance of the next event prediction, and returns the average accuracy and loss.

Computing similarity score:

To verify the similarity between the training and test graphs, the script converts the graphs into adjacency matrices first, and then the similarity score between the matrices is calculated by taking the sum of the differences between the values in the corresponding cells of the two matrices and dividing it by the number of non-zero values in the test matrix. For this, the training matrix is reduced to the test matrix size, and the transitions that are captured during the training process but did not appear during testing are excluded from the calculation. Two cases are considered for the calculation. In Case 1, binary matrices are constructed, where the value of 1 in a cell corresponds to the fact that the transition between the corresponding two activities existed, and the value of 0 represented the fact that there was no transition between the two activities. In Case 2, the matrices are constructed in the same way as in Case 1, except that instead of the value 1, the probability of the transition as predicted by the LSTM model is recorded. After computation, a score of 0 would mean the two graphs are identical, whereas a score of 1 would mean they are completely different. 
