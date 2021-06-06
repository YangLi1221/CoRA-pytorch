import torch
import torch.nn as nn

from .Piecewise_CNN import PCNN
from .Entity_Aware_Embedding import Entity_Aware_Embedding
from .Relation_Augmented_Attention import Relation_Augmented_Attention

class CoRA(nn.Module):

    def __init__(self, args, word_embedding):
        super(CoRA, self).__init__()
        self.hidden_size = args.hidden_size * 3
        self.embedding = Entity_Aware_Embedding(args, word_embedding)
        self.extractor = PCNN(args)
        self.CoRA = Relation_Augmented_Attention(args)
        self.fc_sa = nn.Linear(self.hidden_size * 3, 1)
        self.dropout = nn.Dropout(args.drop_prob)
        self.clf = nn.Linear(self.hidden_size * 3, args.hier3_num_classes)


    def forward(self, scope, sentence, pos1, pos2, piece_mask, en1, en2, bag_rel):
        # Embedding
        x_embedded = self.embedding(sentence, pos1, pos2, en1, en2)
        # Extractor
        x_pcnn = self.extractor(x_embedded, piece_mask)
        # sentence_representation into CoRA
        hier1_pred, x_hier1, hier2_pred, x_hier2, hier3_pred, x_hier3 = self.CoRA(x_pcnn)
        x_coraed = torch.cat([x_hier1, x_hier2, x_hier3], 1)
        x_alpha = self.fc_sa(x_coraed) #bs, 1
        # multi_instance framework
        x_bag = self.multi_instances(x_coraed, x_alpha, scope)
        x_bag = self.dropout(x_bag)
        return_logits = self.clf(x_bag)
        # return the pred
        return return_logits, hier1_pred, hier2_pred, hier3_pred

    def multi_instances(self, x, alpha, scope):
        Bag = []
        for s in scope:
            bag_sents = x[s[0]:s[1]]
            bag_alpha = torch.softmax(alpha[s[0]:s[1]], 0).transpose(0, 1)
            bag_rep = torch.squeeze(torch.mm(bag_alpha, bag_sents))
            Bag.append(bag_rep)
        Bag = torch.stack(Bag)
        return Bag