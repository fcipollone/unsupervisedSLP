import argparse
from models.autoencoder import autoencoder
from models.convolutional_inherited import convolutional_inherited
from models.convolutional_multiplicative_inherited import convolutional_multiplicative_inherited
from models.convolutional_multiplicative_bigger_inherited import convolutional_multiplicative_bigger_inherited
from models.multiplicative_LSTM_rnn_inherited import multiplicative_LSTM_rnn_inherited
from models.multiplicative_LSTM_rnn_bigger_inherited import multiplicative_LSTM_rnn_bigger_inherited
from models.multiplicative_LSTM_rnn_state_classifier_inherited import multiplicative_LSTM_rnn_state_classifier_inherited
from models.rnn_inherited import rnn_inherited
from models.rnn_bigger_inherited import rnn_bigger_inherited
import datetime
from time import gmtime, strftime
import sys


parser = argparse.ArgumentParser(description="Run commands")

parser.add_argument('-time', '--time_length',               type=int,   default=15,               help="Number of timesteps that we train on")
parser.add_argument('-alnr', '--autoencoder_learning_rate', type=float, default=1e-4,             help="Autoencoder learning rate")
parser.add_argument('-clnr', '--classifier_learning_rate',  type=float, default=1e-4,             help="Autoencoder learning rate")
parser.add_argument('-ait', '--iterations_autoencoder',     type=int,   default=10000,            help='number of iterations to run the autoencoder')
parser.add_argument('-cit', '--iterations_classifier',      type=int,   default=10000,            help='number of iterations to run the classifier')
parser.add_argument('-btch', '--batch_size',                type=int,   default=32,               help='size of batch')
parser.add_argument('-clss', '--num_classes',               type=int,   default=7,                help='number of classes')
parser.add_argument('-atr', '--train_autoencoder',          type=bool,  default=True,             help='Whether to train the autoencoder or not')
parser.add_argument('-ctr', '--train_classifier',           type=bool,  default=True,             help='Whether to train the classifier or not')
parser.add_argument('-mdl', '--model_name',                 type=str,   default="rnn_inherited",  help='Whether to train the classifier or not')
parser.add_argument('-vacc', '--validation_accuracy', 		type=bool, 	default=True,				help='Whether to evaluate the accuracy of entire validation set')
parser.add_argument('-tacc', '--test_accuracy', 			type=bool, 	default=True,				help='Whether to evaluate the accuracy of entire test set')
parser.add_argument('-idx', '--indices_list', 					type=str, 	default="1,2", 				help='String list of indices separated by commas')
parser.add_argument('-ctrs', '--train_classifier_small', 	type=bool, 	default=True, 			help='Whether to train the classifier on only a small portion of labeled data')

if __name__ == '__main__':
	# sys.stdout = open('runs.txt', 'a+')
	print "whatever"

	FLAGS = parser.parse_args()

	FLAGS.model_save_dir = '/Users/mila/Documents/Spring2017/CS224s/unsupervisedSLP/saved_models/' #'/Users/frank/stanford/spring2017/slp/project/code/saved_models/'
	FLAGS.load_dir = None #'/Users/frank/stanford/spring2017/slp/project/code/saved_models/convolutional_inherited/autoencoder_and_classifier/convolutional_inherited0_2017_06_06_01_15_20'

	FLAGS.indices = [int(x) for x in FLAGS.indices_list.split(",")]
	FLAGS.num_features = len(FLAGS.indices)

	# feature-numbers, day, 
	model_name = FLAGS.model_name
	datestr = datetime.datetime.now().__str__()
	FLAGS.run_name = model_name + "_".join([str(x) for x in FLAGS.indices]) + '_' + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
	model = None
	if model_name == "rnn_inherited":
		model = rnn_inherited(FLAGS)
	elif model_name == 'rnn_bigger_inherited':
		model = rnn_bigger_inherited(FLAGS)
	elif model_name == 'rnn_autoencoder_inherited':
		model = rnn_autoencoder_inherited(FLAGS)
	elif model_name == "multiplicative_LSTM_rnn_inherited":
		model = multiplicative_LSTM_rnn_inherited(FLAGS)
	elif model_name == "multiplicative_LSTM_rnn_bigger_inherited":
		model = multiplicative_LSTM_rnn_bigger_inherited(FLAGS)
	elif model_name == "multiplicative_LSTM_rnn_state_classifier_inherited":
		model = multiplicative_LSTM_rnn_state_classifier_inherited(FLAGS)
	elif model_name == 'convolutional_inherited':
		model = convolutional_inherited(FLAGS)
	elif model_name == 'convolutional_multiplicative_inherited':
		model = convolutional_multiplicative_inherited(FLAGS)
	elif model_name == 'convolutional_multiplicative_bigger_inherited':
		model = convolutional_multiplicative_bigger_inherited(FLAGS)
	elif model_name == 'autoencoder':
		model = autoencoder(FLAGS)

	model.createModel()
	model.train()

