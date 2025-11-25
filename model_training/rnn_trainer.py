import os
import logging
import pathlib
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BrainToTextDataset, train_test_split_indicies
from omegaconf import OmegaConf

from rnn_model import GRUDecoder

# Load the yaml args
args = OmegaConf.load("rnn_args.yaml")

# Create the output directory
# TODO: Test exist_ok=True
os.makedirs(args["output_dir"], exist_ok=True)


# Set up the logger
logger = logging.getLogger(__name__)

# Remove previous handlers (where to send the log)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Set the logging level to info
logger.setLevel(logging.INFO)

# Setting the format for each log
formatter = logging.Formatter(fmt="%(asctime)s: %(message)s")

# During training, save logs to file in output directory
file_handler = logging.FileHandler(
    str(pathlib.Path(args["output_dir"], "training_log"))
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Prints logs to stdout (standard output file obj)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Use the CPU for training
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Set the seed
np.random.seed(args['seed'])
random.seed(args['seed'])
torch.manual_seed(args['seed'])


# Initialize the GRU model
model = GRUDecoder(
    neural_dim=args["model"]["n_input_features"],
    n_units=args["model"]["n_units"],
    n_days=len(args["dataset"]["sessions"]),
    n_classes=args["dataset"]["n_classes"],
    rnn_dropout=args["model"]["rnn_dropout"],
    input_dropout=args["model"]["input_network"]["input_layer_dropout"],
    n_layers=args["model"]["n_layers"],
    patch_size=args["model"]["patch_size"],
    patch_stride=args["model"]["patch_stride"],
)


# Invoke torch.compile to speed up training
logger.info("Using torch compile...")
torch.compile(model)
logger.info(f"Initialized the {model.__class__.__name__} model")
logger.info(model)

# Log the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Model has {total_params: ,} parameters")

# List of paths to each training file
train_file_paths = [
    os.path.join(args["dataset"]["dataset_dir"], s, "data_train.hdf5")
    for s in args["dataset"]["sessions"]
]

# Ensures there are no duplicates
if len(set(train_file_paths)) != len(train_file_paths):
    raise ValueError("There a duplicate sessions listed in the train dataset")

# Splits trials into train and set sets
# {day: {trials: [1, 2, 3, ..., 100]}}
train_trials, _ = train_test_split_indicies(
    file_paths=train_file_paths,
    test_percentage=0,
    seed=args["dataset"]["seed"],
    bad_trials_dict=None,
)


# Initialize the training dataset
train_dataset = BrainToTextDataset(
    trial_indicies=train_trials,
    split="train",
    days_per_batch=args["dataset"]["days_per_batch"],
    n_batches=args["num_training_batches"],
    batch_size=args["dataset"]["batch_size"],
    must_include_days=None,
    random_seed=args["dataset"]["seed"],
    feature_subset=None,
)

# Efficiently feeds data into GRU in batches
train_loader = DataLoader(
    train_dataset,
    batch_size=None,  # Dataset.__getitem__ already returns batches
    shuffle=args["dataset"]["loader_shuffle"],
    num_workers=args["dataset"]["num_dataloader_workers"],
    pin_memory=True,
)
