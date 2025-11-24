import os
import torch
from torch.utils.data import DataLoader

from dataset import BrainToTextDataset, train_test_split_indicies
from omegaconf import OmegaConf

from rnn_model import GRUDecoder

# Load the yaml args
args = OmegaConf.load("rnn_args.yaml")

# Initialize the GRU class
model = GRUDecoder(
    neural_dim=args['model']['n_input_features'],
    n_units=args['model']['n_units'],
    n_days=len(args['dataset']['sessions']),
    n_classes=args['dataset']['n_classes'],
    rnn_dropout=args['model']['rnn_dropout'],
    input_dropout=args['model']['input_network']['input_layer_dropout'],
    n_layers=args['model']['n_layers'],
    patch_size=args['model']['patch_size'],
    patch_stride=args['model']['patch_stride']
)

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
    batch_size=None, # Dataset.__getitem__ already returns batches
    shuffle=args['dataset']['loader_shuffle'],
    num_workers=args['dataset']['num_dataloader_workers'],
    pin_memory=True
)







