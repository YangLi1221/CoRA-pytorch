import torch
import torch.nn as nn

class Entity_Aware_Embedding(nn.Module):
    def __init__(self, args, word_embedding, lam_pcnn=0.05):
        super(Entity_Aware_Embedding, self).__init__()
        word_embedding = torch.from_numpy(word_embedding)
        self.unk_word_embedding = torch.empty(1, args.word_size)
        nn.init.xavier_uniform_(self.unk_word_embedding)
        blank_word_embedding = torch.zeros(1, args.word_size)
        word_embedding_matrix = torch.cat((word_embedding, self.unk_word_embedding, blank_word_embedding), 0)
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding_matrix, freeze=False, padding_idx=-1)
        self.pos1_embedding = nn.Embedding(2 * args.max_length + 1, args.pos_size)
        self.pos2_embedding = nn.Embedding(2 * args.max_length + 1, args.pos_size)
        self.fc1 = nn.Linear(3 * args.word_size, 3 * args.word_size)
        self.fc2 = nn.Linear(args.word_size + 2 * args.pos_size, 3 * args.word_size)
        self.lam_pcnn = lam_pcnn
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)

    def forward(self, X, X_Pos1, X_Pos2, X_Ent1, X_Ent2):
        X = self.word_embedding(X)
        Xp = self.word_pos_embedding(X, X_Pos1, X_Pos2)
        Xe = self.word_ent_embedding(X, X_Ent1, X_Ent2)
        Xea = self.entity_aware_gate(Xp, Xe, self.lam_pcnn)
        return Xea

    def word_pos_embedding(self, X, X_Pos1, X_Pos2):
        X_Pos1 = self.pos1_embedding(X_Pos1)
        X_Pos2 = self.pos2_embedding(X_Pos2)
        return torch.cat([X, X_Pos1, X_Pos2], -1)

    def word_ent_embedding(self, X, X_Ent1, X_Ent2):
        X_Ent1 = self.word_embedding(X_Ent1).unsqueeze(1).expand(X.shape)
        X_Ent2 = self.word_embedding(X_Ent2).unsqueeze(1).expand(X.shape)
        return torch.cat([X, X_Ent1, X_Ent2], -1)

    def entity_aware_gate(self, Xp, Xe, lam):
        A = torch.sigmoid((self.fc1(Xe / lam)))
        X = A * Xe + (1 - A) * torch.tanh(self.fc2(Xp))
        return X
