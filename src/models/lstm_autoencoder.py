
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_features=47,
        hidden_size=32,
        latent_size=16
    ):
        super().__init__()

        # Encoder: learns the important patterns
        # from the input sequence.
        self.encoder = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Compresses the encoder output.
        self.to_latent = nn.Linear(
            hidden_size,
            latent_size
        )

        # Expands the compressed representation.
        self.from_latent = nn.Linear(
            latent_size,
            hidden_size
        )

        # Decoder: reconstructs the original sequence.
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Produces the reconstructed feature values.
        self.output_layer = nn.Linear(
            hidden_size,
            input_features
        )

    def forward(self, x):

        # x shape: (batch, sequence_length, features)

        # Encode the input sequence.
        _, (hidden, _) = self.encoder(x)

        # Compress the final hidden state.
        latent = self.to_latent(hidden[-1])

        # Expand the compressed representation.
        expanded = self.from_latent(latent)

        # Repeat it across the sequence length.
        repeated = expanded.unsqueeze(1).repeat(
            1, x.size(1), 1
        )

        # Reconstruct the sequence.
        decoded, _ = self.decoder(repeated)

        reconstruction = self.output_layer(decoded)

        return reconstruction
