import argparse
from rnn_inherited import rnn_inherited
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

if __name__ == '__main__':
	FLAGS = parser.parse_args()
	FLAGS.model_save_dir = None    #'/Users/frank/stanford/spring2017/slp/project/code/saved_models/'
	FLAGS.load_dir = None 		   #'/Users/frank/stanford/spring2017/slp/project/code/saved_models/rnn_inherited/autoencoder_and_classifier/2017-06-03 21:59:30.890055,0,rnn_inherited'
	FLAGS.indices = [0]

	# feature-numbers, day, 
	model_name = FLAGS.model_name
	datestr = datetime.datetime.now().__str__()
	FLAGS.run_name = datestr + "-" + '-'.join([str(i) for i in FLAGS.indices]) + "-" + model_name
	model = None
	if model_name == "rnn_inherited":
		model = rnn_inherited(FLAGS)
	elif model_name == "multiplicative_rnn":
		model = multiplicative_LSTM_rnn_inherited(FLAGS)

	model.createModel()
	model.train()

