import torch
import torch.nn as nn
import torch.nn.functional as F
from flashlight.lib.text.decoder import (
    LexiconFreeDecoder,
    LexiconFreeDecoderOptions,
    ZeroLM,
    CriterionType,
)

import string
from typing import Tuple

digits = "0123456789"
letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # O and I removed to avoid confusion with 0 and 1
assert len(letters) == 24, (
    "There should be 24 letters in the alphabet (excluding O and I)"
)
province_codes = [
    "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新",
]  # 31 province codes except Hong Kong, Macau, and Taiwan
assert len(province_codes[0]) == 31, (
    "There should be 31 province codes including Hong Kong, Macau, and Taiwan"
)

default_vocab = ["_"] + list(digits) + list(letters) + list(province_codes[0])


class CRNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, step_width, output_size):
        super(CRNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Adaptive pooling for each level to ensure consistent dimensions
        self.adaptive_pool1 = nn.AdaptiveAvgPool2d((8, 32))
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((4, 16))
        self.adaptive_pool3 = nn.AdaptiveAvgPool2d((2, 8))

        # Channel adjustment layers to balance feature importance
        self.channel_adj1 = nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=1)
        self.channel_adj2 = nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=1)
        self.channel_adj3 = nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=1)

        # Calculate the actual input size for LSTM
        lstm_input_width = 3 * (hidden_size // 2)

        self.lstm = nn.LSTM(
            lstm_input_width, step_width, bidirectional=True, batch_first=True
        )
        # For bidirectional LSTM, hidden size is doubled
        self.fc = nn.Linear(step_width * 2, output_size)

    def forward(self, x):
        # First level features
        x1 = F.relu(self.conv1(x))
        x1_pooled = self.pool(x1)
        x1_out = self.adaptive_pool1(x1_pooled)
        x1_out = self.channel_adj1(x1_out)

        # Second level features
        x2 = F.relu(self.conv2(x1_pooled))
        x2_pooled = self.pool(x2)
        x2_out = self.adaptive_pool2(x2_pooled)
        x2_out = self.channel_adj2(x2_out)

        # Third level features
        x3 = F.relu(self.conv3(x2_pooled))
        x3_pooled = self.pool(x3)
        x3_out = self.adaptive_pool3(x3_pooled)
        x3_out = self.channel_adj3(x3_out)

        # Flatten each feature map
        batch_size = x.size(0)
        x1_flat = x1_out.view(batch_size, -1, x1_out.size(2) * x1_out.size(3))
        x2_flat = x2_out.view(batch_size, -1, x2_out.size(2) * x2_out.size(3))
        x3_flat = x3_out.view(batch_size, -1, x3_out.size(2) * x3_out.size(3))

        # Concatenate along the sequence dimension
        # Each will have shape [batch_size, channels, height*width]
        x_concat = torch.cat([x1_flat, x2_flat, x3_flat], dim=2)

        # Transpose to get [batch_size, sequence_length, features]
        # This format is suitable for the LSTM in the decoder
        x_concat = x_concat.transpose(1, 2)

        # BiLSTM forward pass
        lstm_out, _ = self.lstm(x_concat)
        # Project to output size
        logits = self.fc(lstm_out)

        return logits


class Decoder:
    def __init__(
        self,
        vocab=None,
        blank_id=0,
        beam_width=25,
        beam_size_token=10,
        beam_threshold=20.0,
    ):
        # Define your vocabulary (must match model training)
        # blank token is index 0 by convention for CTC
        if vocab is None:
            vocab = default_vocab

        self.vocab = vocab
        self.blank_id = blank_id
        self.beam_width = beam_width

        # Setup decoder options
        self.decoder_options = LexiconFreeDecoderOptions(
            beam_size=beam_width,
            beam_size_token=beam_size_token,
            beam_threshold=beam_threshold,
            lm_weight=0.0,
            sil_score=0.0,
            log_add=False,
            criterion_type=CriterionType.CTC,
        )

        # Create decoder
        self.decoder = LexiconFreeDecoder(
            self.decoder_options,
            lm=ZeroLM(),  # No language model
            sil_token_idx=-1,  # No silence token in CTC
            blank_token_idx=blank_id,
            transitions=[],  # No transitions for CTC
        )

    def decode(self, logits: torch.Tensor) -> str:
        # Convert logits to log probabilities
        log_probs = F.log_softmax(logits, dim=2)

        # Flashlight expects emissions in format [Time, Batch, Vocab]
        # Current format is [Batch, Time, Vocab]
        emissions = log_probs.transpose(0, 1).contiguous()

        # Decode for each item in batch (assuming batch_size = 1)
        batch_size = emissions.size(1)
        results = []

        for b in range(batch_size):
            # Get emissions for this batch item
            batch_emissions = emissions[:, b, :].cpu()

            # Decode
            decoded = self.decoder.decode(
                batch_emissions.data_ptr(),
                batch_emissions.size(0),
                batch_emissions.size(1),
            )

            if decoded:
                # Get best hypothesis
                best_hyp = decoded[0]
                # Convert token indices to string
                tokens = []
                for token_idx in best_hyp.tokens:
                    if token_idx < len(self.vocab):
                        tokens.append(self.vocab[token_idx])
                result = "".join(tokens)
                results.append(result)
            else:
                results.append("")

        return results[0] if results else ""


class Reco(nn.Module):
    def __init__(self, input_size, hidden_size, step_width, output_size, vocab=None):
        super(Reco, self).__init__()
        self.encoder = CRNNEncoder(input_size, hidden_size, step_width, output_size)
        self.decoder = Decoder(vocab=vocab)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, str]:
        logits = self.encoder(x)
        decoded_str = self.decoder.decode(logits)
        return logits, decoded_str
