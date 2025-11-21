import os
import torch
from torch.utils.data import DataLoader

from dataset import BrainToTextDataset, train_test_split_indicies
from omegaconf import OmegaConf

# Load the yaml args
args = OmegaConf.load("rnn_args.yaml")

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

# Determine if a subset of features should be used
feature_subset = None

# Initialize the training dataset
train_dataset = BrainToTextDataset(
    trial_indicies=train_trials,
    split="train",
    days_per_batch=args["dataset"]["days_per_batch"],
    n_batches=args["num_training_batches"],
    batch_size=args["dataset"]["batch_size"],
    must_include_days=None,
    random_seed=args["dataset"]["seed"],
    feature_subset=feature_subset,
)

# TODO: what is this?
train_loader = DataLoader(
    train_dataset,
    batch_size=None,
    shuffle=args['dataset']['loader_shuffle'],
    num_workers=args['dataset']['num_dataloader_workers'],
    pin_memory=True
)


