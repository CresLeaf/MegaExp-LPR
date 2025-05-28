import torch
import torch.nn as nn
import torch.nn.functional as F
import k2
import string
from typing import Tuple


class CNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CNNEncoder, self).__init__()
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
        self.channel_adj3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

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

        return x_concat


def reference_fst() -> Tuple[k2.Fsa, dict]:
    # Create character to index mapping
    # Start with blank token and special characters
    character_map = {
        "<blank>": 0,
    }

    # Add all possible characters from the pattern
    all_chars = set()
    all_chars.update(["京", "津", "冀"])  # TODO: Add more provinces
    all_chars.update(string.ascii_uppercase)
    all_chars.update(string.digits)
    all_chars.add("\n")

    # Map characters to indices (starting from 1 since 0 is blank)
    for idx, char in enumerate(sorted(all_chars), start=1):
        character_map[char] = idx

    # Build FSA using k2
    # States: 0=start, 1=after province, 2=after letter, 3=optional newline,
    # 4-9=after 1-6 alphanumerics, 10=final
    arcs = []

    # Start state
    for province in ["京", "津", "冀"]:
        arcs.append(
            [0, 1, character_map[province], 0]
        )  # From state 0 to state 1 with province char

    # After province, expect a letter
    for letter in string.ascii_uppercase:
        arcs.append(
            [1, 2, character_map[letter], 0]
        )  # From state 1 to state 2 with uppercase letter

    # After letter, optional newline
    arcs.append([2, 3, character_map.get("\n", 0), 0])  # Optional newline
    arcs.append([2, 4, 0, 0])  # Skip newline, epsilon transition

    # From newline state, start alphanumeric sequence
    for char in string.ascii_uppercase + string.digits:
        arcs.append([3, 4, character_map[char], 0])  # First alphanumeric

    # Alphanumeric sequence states (5-6 characters)
    for i in range(4, 9):  # States 4-8 (for positions 1-5)
        for char in string.ascii_uppercase + string.digits:
            arcs.append([i, i + 1, character_map[char], 0])  # Next alphanumeric

    # Make states 9 and 10 final (5 or 6 characters)
    arcs.append([9, 10, 0, 0])  # Epsilon to final state
    arcs.append([10, -1, -1, 0])  # Final state marker

    # Convert to k2 format
    arcs_tensor = torch.tensor(arcs, dtype=torch.int32)
    fsa = k2.Fsa.from_tensor(arcs_tensor)

    # Add necessary properties
    fsa.requires_grad_(False)
    fsa = k2.arc_sort(fsa)

    return fsa, character_map


class RegexConstrainedCTCDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegexConstrainedCTCDecoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        # For bidirectional LSTM, hidden size is doubled
        self.fc = nn.Linear(hidden_size * 2, output_size)

        # Create character-to-index mapping
        self.output_size = output_size
        self.blank_idx = 0  # CTC blank token index
        self.constraint_fsa, self.alphabet = reference_fst()

    def forward(self, x):
        # BiLSTM forward pass
        lstm_out, _ = self.lstm(x)
        # Project to output size
        logits = self.fc(lstm_out)
        return logits

    def decode(self, logits, beam_size=5):
        """
        Decode using constrained CTC prefix beam search
        """
        # Apply log softmax to get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Convert to dense log probs for k2 (batch size 1 for simplicity)
        batch_size = log_probs.size(0)
        results = []

        for i in range(batch_size):
            sequence_log_probs = log_probs[i].cpu()

            # Create a lattice/FSA from the log probs
            supervision = k2.SupervisionSegment(0, 0, sequence_log_probs.size(0))
            supervision_segments = torch.tensor(
                [[0, 0, sequence_log_probs.size(0)]], dtype=torch.int32
            )

            # Convert log probs to dense FSA
            dense_fsa = k2.DenseFsaVec(
                sequence_log_probs.unsqueeze(0), supervision_segments
            )

            # If we have a constraint, apply it
            if self.constraint_fsa is not None:
                # Intersect dense FSA with constraint FSA
                lattice = k2.intersect_dense(
                    dense_fsa, self.constraint_fsa, output_beam=beam_size
                )

                # Find the best path
                best_path = k2.shortest_path(lattice, use_double_scores=True)

                # Convert path to labels, removing blanks and duplicates
                labels = []
                prev = -1
                for arc in best_path.arcs.values():
                    label = arc.label.item()
                    if label != 0 and label != prev:  # Skip blanks and duplicates
                        labels.append(label)
                    prev = label

                # Convert indices back to characters
                idx_to_char = {v: k for k, v in self.alphabet.items()}
                text = "".join(idx_to_char.get(l, "") for l in labels)
                results.append(text)
            else:
                # Regular CTC decoding without constraints
                # ...
                pass

        return results
