import torch
import torch.nn as nn


class PCNN(nn.Module):
    def __init__(self, args):
        super(PCNN, self).__init__()
        mask_embedding = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.mask_embedding = nn.Embedding.from_pretrained(mask_embedding, freeze=True)
        self.cnn = nn.Conv1d(3 * args.word_size, args.hidden_size, 3, padding=1)
        self.hidden_size = args.hidden_size * 3
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.cnn.weight)
        nn.init.zeros_(self.cnn.bias)

    def forward(self, Xea, X_mask):
        Xea = Xea.transpose(1, 2)
        X = self.cnn(Xea)
        X = X.transpose(1, 2)
        X = self.pool(X, X_mask)
        X = torch.tanh(X)
        return X

    def pool(self, X, X_mask):
        X_mask = self.mask_embedding(X_mask)
        X = torch.max(torch.unsqueeze(X_mask, 2) * torch.unsqueeze(X, 3), 1)[0]
        return X.view(-1, self.hidden_size)
