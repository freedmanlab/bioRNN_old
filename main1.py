import os, sys
import tensorflow as tf
from model import build_model
from train_supervised import train_loop
from parameters import par

# Match GPU IDs to nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0 and len(sys.argv) > 1:
    tf.config.experimental.set_visible_devices(gpus[int(sys.argv[1])], 'GPU')

n_hidden = 256
model = build_model(par['n_input'], n_hidden, par['n_output'], connection_prob=0.8)
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_loop(model, opt, 'dmc', 2000)
