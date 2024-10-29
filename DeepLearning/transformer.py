import random
import unittest
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from vocabulary import Vocabulary
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from utils import construct_future_mask


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,  # Dimension of the hidden states
        ff_dim: int,      # Dimension of the feedforward layer
        num_heads: int,   # Number of attention heads
        num_layers: int,  # Number of encoder/decoder layers
        max_decoding_length: int,  # Maximum length for decoding
        vocab_size: int,  # Size of the vocabulary
        padding_idx: int, # Padding index for the embeddings
        bos_idx: int,     # Index for the beginning-of-sequence token
        dropout_p: float, # Dropout probability
        tie_output_to_embedding: Optional[bool] = None,  # Option to tie output weights to input embeddings
    ):
        super().__init__()
        # Initialize the embedding layer for input and output
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        
        # Initialize the transformer encoder
        self.encoder = TransformerEncoder(
            self.embed, hidden_dim, ff_dim, num_heads, num_layers, dropout_p
        )
        
        # Initialize the transformer decoder
        self.decoder = TransformerDecoder(
            self.embed,
            hidden_dim,
            ff_dim,
            num_heads,
            num_layers,
            vocab_size,
            dropout_p,
            tie_output_to_embedding,
        )

        # Store parameters for padding and beginning-of-sequence token
        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.max_decoding_length = max_decoding_length
        self.hidden_dim = hidden_dim
        
        # Reset parameters using Xavier initialization
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize parameters for layers with dimension > 1 using Xavier uniform distribution
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TestTransformer(unittest.TestCase):
    def test_transformer_inference(self):
        # Set a random seed for reproducibility
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create a dummy corpus and build a vocabulary from it
        corpus = [
            "Hello my name is Joris and I was born with the name Joris.",
            "Dit is een Nederlandse zin.",
        ]
        en_vocab = Vocabulary(corpus)
        en_vocab_size = len(en_vocab.token2index.items())
        
        # Perform inference without gradient calculation
        with torch.no_grad():
            # Initialize the transformer model
            transformer = Transformer(
                hidden_dim=512,
                ff_dim=2048,
                num_heads=8,
                num_layers=6,
                max_decoding_length=10,
                vocab_size=en_vocab_size,
                padding_idx=en_vocab.token2index[en_vocab.PAD],
                bos_idx=en_vocab.token2index[en_vocab.BOS],
                dropout_p=0.1,
                tie_output_to_embedding=True,
            )
            transformer.eval()  # Set the model to evaluation mode

            # Prepare input for the encoder and create a mask for padding
            encoder_input = torch.IntTensor(
                en_vocab.batch_encode(corpus, add_special_tokens=False)
            )
            src_padding_mask = encoder_input != transformer.padding_idx
            
            # Get encoder output
            encoder_output = transformer.encoder.forward(
                encoder_input, src_padding_mask=src_padding_mask
            )
            # Check for NaN values in the encoder output
            self.assertEqual(torch.any(torch.isnan(encoder_output)), False)

            # Prepare initial input for the decoder (start with BOS token)
            decoder_input = torch.IntTensor(
                [[transformer.bos_idx], [transformer.bos_idx]]
            )
            future_mask = construct_future_mask(seq_len=1)  # Create initial future mask
            
            # Decode step by step
            for i in range(transformer.max_decoding_length):
                # Get decoder output
                decoder_output = transformer.decoder(
                    decoder_input,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask,
                )
                # Predict the next token by taking argmax of the last token's output
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)

                # Update decoder input with the predicted token
                decoder_input = torch.cat((decoder_input, predicted_tokens), dim=-1)
                future_mask = construct_future_mask(decoder_input.shape[1])  # Update future mask

        # Ensure the decoder output shape is as expected
        self.assertEqual(decoder_input.shape, (2, transformer.max_decoding_length + 1))
        # Ensure that the decoder input contains at least one token other than the BOS token
        self.assertEqual(torch.all(decoder_input == transformer.bos_idx), False)


if __name__ == "__main__":
    unittest.main()  # Run the unit tests
