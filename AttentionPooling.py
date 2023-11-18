import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, token_embeddings, mask):
        # Compute attention scores
        attention_scores = torch.matmul(token_embeddings, self.attention_weights).squeeze(-1)  # (batch_size, max_seq_len)
        attention_scores = attention_scores.to(torch.float32)  # Convert to float32
        mask = mask.to(torch.float32)  # Convert to float32
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)


        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(1)  # (batch_size, 1, max_seq_len)

        # Compute weighted sum of token embeddings
        pooled_output = torch.bmm(attention_weights, token_embeddings).squeeze(1)  # (batch_size, hidden_size)
        return pooled_output
