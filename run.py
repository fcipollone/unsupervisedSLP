import argparse
from rnn_inherited import rnn_inherited
<<<<<<< HEAD
=======
from convolutional_inherited import convolutional_inherited
>>>>>>> origin/master
from multiplicativeLSTM_rnn_inherited import multiplicative_LSTM_rnn_inherited
import datetime

parser = argparse.ArgumentParser(description="Run commands")

parser.add_argument('-time', '--time_length',               type=int,   default=15,               help="Number of timesteps that we train on")
parser.add_argument('-alnr', '--autoencoder_learning_rate', type=float, default=1e-4,             help="Autoencoder learning rate")
parser.add_argument('-clnr', '--classifier_learning_rate',  type=float, default=1e-4,             help="Autoencoder learning rate")
parser.add_argument('-ait', '--iterations_autoencoder',     type=int,   default=1000,            help='number of iterations to run the autoencoder')
parser.add_argument('-cit', '--iterations_classifier',      type=int,   default=1000,            help='number of iterations to run the classifier')
parser.add_argument('-btch', '--batch_size',                type=int,   default=32,               help='size of batch')
parser.add_argument('-clss', '--num_classes',               type=int,   default=7,                help='number of classes')
parser.add_argument('-atr', '--train_autoencoder',          type=bool,  default=True,             help='Whether to train the autoencoder or not')
parser.add_argument('-ctr', '--train_classifier',           type=bool,  default=True,             help='Whether to train the classifier or not')
parser.add_argument('-mdl', '--model_name',                 type=str,   default="rnn_inherited",  help='Whether to train the classifier or not')
parser.add_argument('-vacc', '--validation_accuracy', 		type=bool, 	default=True,				help='Whether to evaluate the accuracy of entire validation set')
parser.add_argument('-tacc', '--test_accuracy', 			type=bool, 	default=False,				help='Whether to evaluate the accuracy of entire test set')


if __name__ == '__main__':
	FLAGS = parser.parse_args()
	FLAGS.model_save_dir = '/Users/mila/Documents/Spring2017/CS224s/unsupervisedSLP/saved_models/'    #'/Users/frank/stanford/spring2017/slp/project/code/saved_models/'
	FLAGS.load_dir = None #'/Users/mila/Documents/Spring2017/CS224s/unsupervisedSLP/saved_models/2017-06-05 16/52/37.780809-0-multiplicative_rnn' #None 		   #'/Users/frank/stanford/spring2017/slp/project/code/saved_models/rnn_inherited/autoencoder_and_classifier/2017-06-03 21:59:30.890055,0,rnn_inherited'
	FLAGS.indices = [0]
	FLAGS.num_features = len(FLAGS.indices)

	# feature-numbers, day, 
	model_name = FLAGS.model_name
	datestr = datetime.datetime.now().__str__()
	FLAGS.run_name = model_name + "_".join([str(x) for x in self.data.indices]) + '_' + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
	model = None
	if model_name == "rnn_inherited":
		model = rnn_inherited(FLAGS)
	elif model_name == "multiplicative_rnn":
		model = multiplicative_LSTM_rnn_inherited(FLAGS)
	elif model_name == 'convolutional_inherited':
		model = convolutional_inherited(FLAGS)
	elif model_name == 'rnn_bigger_inherited':
		model_

	model.createModel()
	model.train()


