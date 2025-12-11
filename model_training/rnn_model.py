import torch
from torch import nn


class GRUDecoder(nn.Module):
    """
    Defines the GRU RNN decoder.
    """

    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=5,
        patch_size=0,
        patch_stride=0,
    ):
        """
        Attributes
        ----------
        neural_dim     (int)        - number of channels in a single timestep (e.g. 512)
        n_units        (int)        - number of hidden units in each recurrent layer
        n_days         (int)        - number of days in the dataset
        n_classes      (int)        - number of classes
        rnn_dropout    (float)      - percentage of units to dropout during training
        input_dropout  (float)      - percentage of input units to dropout before training
        n_layers       (int)        - number of recurrent layers
        patch_size     (int)        - number of timesteps to concat on initial input layer
        patch_stride   (int)        - number of timesteps to stride over when concatenating initial input layer
        """
        # Initializing the parent class inside child class
        super(GRUDecoder, self).__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout

        self.n_layers = n_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Output between -1 and 1
        # "A shallower tanh"
        self.day_layer_activation = nn.Softsign()

        # Creates an identity matrix of size 512x512 for each day
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )

        # Creates a zeros matrix of size 1x512 for each day
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        # Prevents co-adaptation of feature detectors
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim

        # For "strided inputs", the input size of the first recurrent layer is
        # input size * patch size (e.g. 512 * 14)
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # Create the GRU object instance
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            batch_first=True,  # The first dimension for our input is the batch dim
            bidirectional=False,
        )

        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                # Rotates each hidden layer's gate's weight matrices
                # (e.g. 128x128) so that each row is perpendicular from
                # every other row
                # true for rows and columns
                # Gives the model something to learn
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                # Keeps variance of output the same as the variance of input
                # Helps solve the exploding/vanishing gradient problem
                nn.init.xavier_uniform_(param)

        # Output weight matrix: Classes x Units (e.g. 41x128)
        self.out = nn.Linear(self.n_units, self.n_classes)
        # Xavier initialization of weights
        nn.init.xavier_uniform_(self.out.weight)

        # Initialize h0 hidden state vector with shape
        # (layer, batch, recurrent units), (e.g. 1, 1, 128)
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states=None, return_state=False):
        """

        Parameters
        ----------
        x               (tensor)    - batch of trials of shape: (batch_size, time_series_length, neural_dim)
        day_idx         (tensor)    - list of day indices corresponding to the day of the trial in the batch x.
        states          (bool)      - None if initial hidden states are not yet determined
        return_state    (bool)      - True if hidden states are to be returned

        Returns
        -------
        TODO: Finish this return docstring
        logits
        """

        # Stacks separate weight matrices for each day
        # with shape (days, neural_dim, neural dim) (e.g. 45, 512, 512)
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)

        # Concatenates individual bias vectors for each day
        # shape (days, 1, 512)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(
            1
        )

        # d: neural features, t: time step, b: batch, k: output feature dim
        # For each batch, time step (b, t), multiply neural features by weight matrix (d x k)
        # Sum over neural features (d)
        # Add biases (batch, output feature dim)
        # producing output (batch, time step, output feature dim)
        # Final output: transformed neural sequences (e.g. 512)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        # Output is between -1 and 1
        x = self.day_layer_activation(x)

        # Apply dropout to the output of the day specific layer
        # (e.g. 20% of 512 input features)
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # Since patch size is 14, perform input concat operation
        if self.patch_size > 0:

            x = x.unsqueeze(1)              # [batches, 1, timesteps, feature_dim]

            #Pput timesteps in the last slot because unfold works on the last dim
            x = x.permute(0, 3, 1, 2)       # [batches, feature_dim, 1, timesteps]

            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dum, num_patches, patch_size]

            # Remove dummy height dimensions and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)          # [batches, num_patches, patch_size, feature_dim]