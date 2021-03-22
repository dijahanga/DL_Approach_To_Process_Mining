

Following is the supplementary material for the article “A Graph-Based Approach to Interpreting Recurrent Neural Networks in Process Mining” by K. M. Hanga, Y. Kovalchuk, and M. M. Gaber.
The code provided in this repository can be used to perform the following tasks:

•	Prediction of the next activity to be executed in a process instance.

•	Prediction of the continuation of a process instance (i.e. its suffix).

•	Generating a process model graph explaining the decision-making of the LSTM model when predicting process event sequences. 

•	Computing the similarity between the graphs to validate the generalising ability of the model.

•	Perform some process mining tasks.

The scripts train a Long Short Term Memory (LSTM)-based predictive model using historical data (i.e. completed process instances). The models are evaluated on test datasets.

Requirements:

Python 3. Additionally, the following Python libraries are required to run the code: Keras (as a backend, TensorFlow, or Theano is needed), unicodecsv, numpy, sklearn, matplotlib, pydot, graphviz, h5py. (Latest versions might be mandatory).
 
Data format:

The tool assumes the input is a complete log of all traces in the CSV format which has a case ID column, and an event column containing activity names or ID. The input log is in some cases split into 70% (training set) and 30% (test set). Sample datasets used in the paper are provided in the data folder.

