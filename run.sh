#!/bin/bash

#Features:
# Multiplicative LSTM 10000, classifier for 3000

# Architectures
# Basic inherited
# RNN
# Autoencoder trained for 1000, 10000, Classifier 3000
# Multiplicative LSTM
# Autoencoder trained for 1000, 10000, Classifier 3000
# Add convolutional layer 
# Multiplicative LSTM
# Autoencoder trained for 1000, 10000, Classifier 3000
# Typical Autoencoder
# Different hyperparameters on the best one

# Number of hidden layers = 100
# Could change them

# multiplicative rnn features
# python run.py -ait 1 -cit 1 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,3
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,4
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,5
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,6
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,7
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,8
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,9
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,10
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,11
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,12
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,13
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,14
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,15
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,16
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,17
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,18
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,19
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,20

# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 21,22,23,24,25,26,27,28,29,30,31,32
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,3
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20

# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 8,9,10,11,12,13,14,15,16,17,18,19,20


# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 8,9,10,11,12,13,14,15,16,17,18,19,20

#This was around 0.25 accuracy
# python run.py -ait 10000 -cit 3000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20

# does not learn 
# python run.py -ait 5000 -cit 10000 -mdl multiplicative_LSTM_rnn_inherited -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20
python run.py -ait 5000 -cit 10000 -mdl multiplicative_LSTM_rnn_bigger_inherited -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20
python run.py -ait 5000 -cit 10000 -mdl multiplicative_LSTM_rnn_state_classifier_inherited -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20
python run.py -ait 5000 -cit 10000 -mdl convolutional_multiplicative_inherited -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20
python run.py -ait 5000 -cit 10000 -mdl convolutional_multiplicative_bigger_inherited -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20
python run.py -ait 5000 -cit 10000 -mdl rnn_inherited -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20
python run.py -ait 5000 -cit 10000 -mdl rnn_bigger_inherited -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20
python run.py -ait 5000 -cit 10000 -mdl autoencoder -idx 1,2,5,7,8,9,11,12,13,14,15,16,17,18,19,20


# if model_name == "rnn_inherited":
# 		model = rnn_inherited(FLAGS)
# 	elif model_name == 'rnn_bigger_inherited':
# 		model = rnn_bigger_inherited(FLAGS)
# 	elif model_name == 'rnn_autoencoder_inherited':
# 		model = rnn_autoencoder_inherited(FLAGS)
# 	elif model_name == "multiplicative_LSTM_rnn_inherited":
# 		model = multiplicative_LSTM_rnn_inherited(FLAGS)
# 	elif model_name == "multiplicative_LSTM_rnn_bigger_inherited":
# 		model = multiplicative_LSTM_rnn_bigger_inherited(FLAGS)
# 	elif model_name == "multiplicative_LSTM_rnn_state_classifier_inherited":
# 		model = multiplicative_LSTM_rnn_state_classifier_inherited(FLAGS)
# 	elif model_name == 'convolutional_inherited':
# 		model = convolutional_inherited(FLAGS)
# 	elif model_name == 'convolutional_multiplicative_inherited':
# 		model = convolutional_multiplicative_inherited(FLAGS)
# 	elif model_name == 'convolutional_multiplicative_bigger_inherited':
# 		model = convolutional_multiplicative_bigger_inherited(FLAGS)
# 	elif model_name == 'autoencoder':
# 		model = autoencoder(FLAGS)




# different architectures







