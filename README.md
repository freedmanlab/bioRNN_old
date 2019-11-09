# bioRNN

A Tensorflow 2 implementation of the RNN described in the methods section of [Circuit mechanisms for the maintenance and manipulation of working memory](https://www.nature.com/articles/s41593-019-0414-3). You'll find the underlying equations there. The Tensorflow 1 implementation is at [Short-term-plasticity-RNN](https://github.com/freedmanlab/Short-term-plasticity-RNN). 

## Code Structure

- **bioRNN cell** is found in `layers.py`. We've implemented the RNN cell as a Keras layer for maximum modularity and ease of use - you can plug it into any Keras model and training method, and it'll just work. 
- **The model** is found in `model.py`. It builds on bioRNN cell. Here, you'll find two functions. `build_model` is a function which will return a bioRNN model (with the given parameters) in the form of a Keras model. This model takes as input a single time-step of observation, and returns logits over actions as well as a list of hidden states to be maintained. The second function you'll find in `model.py` is `do_trial`, which takes a model as described above and loops over a trial, described by a tensor of shape (Tsteps, Batch, n_input).
- **The training method** is found in `loss.py` and `train_supervised.py`. The supervised loss implemented in `loss.py` is described in the method section of the original paper.
- **The task data** is produced by `parameters.py` and `stimulus.py`. Note that `parameters.py` is not relevant to any of the stuff above. This code was taken from the [Context dependent gating](https://github.com/freedmanlab/Context-Dependent-Gating) repo. 

## Tests

A few basic sanity checks have been done, and the network trains up, but more tests and a thorough review and probably needed before declaring this implmentation "correct."
