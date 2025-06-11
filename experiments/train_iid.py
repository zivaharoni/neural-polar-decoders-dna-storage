#%% imports
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import numpy as np
import tensorflow as tf
# Disable XLA JIT to avoid unsupported RaggedTensor operations on GPU
tf.config.optimizer.set_jit(False)
import wandb
from wandb.integration.keras import WandbMetricsLogger
from keras.optimizers import Adam
from src.generators import info_bits_generator
from src.builders import build_neural_polar_decoder_iid as model_builder
from src.channels import InsertionDeletionSubstitutionGallager
from src.callbacks import ReduceLROnPlateauCustom, SaveModelCallback
from src.utils import (save_args_to_json, load_json, print_config_summary, gpu_init, safe_wandb_init)

#%% set configurations
print(f"TF version: {tf.__version__}")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_init(allow_growth=True)

eager_mode = False
if eager_mode:
    print("Running in eager mode")
    tf.config.run_functions_eagerly(True)

def get_args():
    parser = argparse.ArgumentParser(description="Train or evaluate Neural Polar Decoder.")
    parser.add_argument("--channel", type=str, choices=["deletion", "ids"], default="deletion",
                        help="Channel type for data generation.")
    parser.add_argument("--i", type=float, default=0.0, 
                        help="insertion rate.")
    parser.add_argument("--d", type=float, default=0.1, 
                        help="deletion rate.")
    parser.add_argument("--s", type=float, default=0.0, 
                        help="substitution rate.")
    parser.add_argument("--batch", type=int, default=10,
                        help="Batch size.")
    parser.add_argument("--N", type=int, default=64,
                        help="Block length.")
    parser.add_argument("--lr", type=float, default=None, 
                        help="lr for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--steps_per_epoch", type=int, default=5000,
                        help="Number of steps per training epoch.")
    parser.add_argument("--mc_length", type=int, default=1000,
                        help="MC length used for evaluation.")
    parser.add_argument("--load_path", type=str, default="", 
                        help="path to intial weights to be loaded.")
    parser.add_argument("--embedding_config_path", type=str, default="./configs/attention_small_embedding.json",
                        help="embedding function.")
    parser.add_argument("--npd_config_path", type=str, default="./configs/npd_small_config.json",
                        help="Path to npd configs.")
    parser.add_argument("--optimizer_config_path", type=str, default="./configs/optimizer_config.json",
                        help="Path to optimizer configs.")
    parser.add_argument("--save_dir_path", type=str, default="./results/train-iid-deletion",
                        help="Path to save trained model weights.")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1,
                        help="Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.")
    parser.add_argument("--wandb_mode", type=str, choices=["online", "dryrun"], default="online",
                        help="WandB mode.")
    return parser.parse_args()

args = get_args()
save_args_to_json(args, os.path.join(args.save_dir_path, "args.json"))
embedding_config = load_json(args.embedding_config_path)
npd_config = load_json(args.npd_config_path)
optimizer_config = load_json(args.optimizer_config_path)
if args.lr is not None:
    optimizer_config["learning_rate"] = args.lr

os.makedirs(args.save_dir_path, exist_ok=True)
os.makedirs( os.path.join(args.save_dir_path, 'model'), exist_ok=True)
model_path = os.path.join(args.save_dir_path, 'model', "model.weights.h5")
print(f"Model path: {model_path}")

safe_wandb_init(project="npds_dna_publish",
                entity="data-driven-polar-codes",
                tags=["train", "iid", "deletion"],
                mode=args.wandb_mode,
                config=dict(**vars(args),**embedding_config, **npd_config, **optimizer_config))

#%% Print the model configuration

print_config_summary(vars(args), title="Args")
print_config_summary(embedding_config, title="Embedding NN")
print_config_summary(npd_config, title="Neural Polar Decoder")
print_config_summary(optimizer_config, title="Optimizer")

#%%  Here the channel can be changed to desired one
if args.channel == "deletion":
    channel = InsertionDeletionSubstitutionGallager(i=0.0, d=args.d, s=0.0, pad_symbol=2, pad_length=args.N)    
    input_shape=(
    (args.batch, args.N, 1),  # shape of x
    (args.batch, args.N, 1)   # shape of y
)
elif args.channel == "ids":
    pad_length = int(1.5 * args.N)
    channel = InsertionDeletionSubstitutionGallager(i=args.i, d=args.d, s=args.s, pad_symbol=2, pad_length=pad_length)    
    input_shape=(
    (args.batch, args.N, 1),  # shape of x
    (args.batch, pad_length, 1)   # shape of y
)
else:
    raise ValueError(f"Invalid channel type: {args.channel}. Choose 'deletion'.")

#%% Create training dataset
data_gen = info_bits_generator(args.batch, args.N)
def channel_mapping(x):
    x = tf.expand_dims(x, -1)
    return x, channel(x)

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_gen,
    output_signature=tf.TensorSpec(shape=(args.batch, args.N), dtype=tf.int32)
).prefetch(tf.data.AUTOTUNE).map(channel_mapping).map(lambda x, y: (tf.ensure_shape(x, input_shape[0]), tf.ensure_shape(y, input_shape[1])) )

#%% Build the model


npd = model_builder(embedding_config, npd_config, input_shape, load_path=args.load_path if len(args.load_path) > 0 else None)
npd.compile(optimizer=Adam(learning_rate=optimizer_config["learning_rate"],
                           beta_1=optimizer_config["beta_1"],
                           beta_2=optimizer_config["beta_2"],
                           clipnorm=1.0))


#%% Train the model
lr_scheduler = ReduceLROnPlateauCustom(monitor='mi', factor=optimizer_config["factor"],
                                       patience=optimizer_config["patience"], verbose=args.verbose,
                                       min_lr=optimizer_config["min_lr"], mode=optimizer_config["mode"])
save_callback = SaveModelCallback(save_path=model_path, save_freq=1)

history = npd.fit(train_dataset,
                  epochs=args.epochs,
                  steps_per_epoch=args.steps_per_epoch,
                  callbacks=[lr_scheduler, WandbMetricsLogger(), save_callback], verbose=args.verbose)
print("Training complete.")


#%% Save model weights
npd.save_weights(model_path)  # creates

print("Training complete. Model saved to:", model_path)
