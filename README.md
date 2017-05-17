# unsupervisedSLP

The structure of the data is set up such that there are four directories which will store the data:
'DC', 'JE', 'JK', 'KL'

Inside each is a bash file called getData.sh, these should be called within each directory to download all the data into their respective locations.

The first pass at the RNN is in rnn.py

The goal here is to train something that will improve on the initial loss a good amount, and we're not there yet but at least we have the data reader written. I think for our milestone we'll be good if we have some sort of RNN working. We could also try a different dataset.
