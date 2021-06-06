import torch
import torch.nn as nn

class Relation_Augmented_Attention(nn.Module):

    def __init__(self, args):
        super(Relation_Augmented_Attention, self).__init__()
        self.hidden_size = args.hidden_size * 3
        self.hier1_relation_matrix = nn.Linear(args.hier1_num_classes, self.hidden_size, bias=False)
        self.hier2_relation_matrix = nn.Linear(args.hier2_num_classes, self.hidden_size, bias=False)
        self.hier3_relation_matrix = nn.Linear(args.hier3_num_classes, self.hidden_size, bias=False)

        self.hier1_fc1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.hier1_fc2 = nn.Linear(self.hidden_size, 1024)
        self.hier1_fc3 = nn.Linear(1024, self.hidden_size)
        self.hier1_layer_norm = nn.LayerNorm(self.hidden_size)

        self.hier2_fc1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.hier2_fc2 = nn.Linear(self.hidden_size, 1024)
        self.hier2_fc3 = nn.Linear(1024, self.hidden_size)
        self.hier2_layer_norm = nn.LayerNorm(self.hidden_size)

        self.hier3_fc1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.hier3_fc2 = nn.Linear(self.hidden_size, 1024)
        self.hier3_fc3 = nn.Linear(1024, self.hidden_size)
        self.hier3_layer_norm = nn.LayerNorm(self.hidden_size)

        self.init_weight()

    def init_weight(self):
        nn.init.orthogonal_(self.hier1_relation_matrix.weight)
        nn.init.orthogonal_(self.hier2_relation_matrix.weight)
        nn.init.orthogonal_(self.hier3_relation_matrix.weight)


    def forward(self, x): # x --> bs, hs

        hier1_rel_logits = torch.mm(x, self.hier1_relation_matrix.weight) #->(bs, rl)
        hier1_rel_index = torch.softmax(hier1_rel_logits, -1) # on the rl dim
        hier1_re_rel = torch.mm(hier1_rel_index, self.hier1_relation_matrix.weight.transpose(0,1)) # bs, hs
        # gate
        hier1_concat_rel = torch.cat((x, hier1_re_rel), 1) # bs, 2hs
        hier1_gate_rel = torch.sigmoid(self.hier1_fc1(hier1_concat_rel))
        hier1_context_rel = hier1_gate_rel * x + (1 - hier1_gate_rel) * hier1_re_rel
        # MLP linear
        hier1_output_rel = self.hier1_fc3(torch.relu(self.hier1_fc2(hier1_context_rel)))
        # add & norm
        hier1_output_rel += x
        hier1_output_rel = self.hier1_layer_norm(hier1_output_rel)

        hier2_rel_logits = torch.mm(x, self.hier2_relation_matrix.weight)  # ->(bs, rl)
        hier2_rel_index = torch.softmax(hier2_rel_logits, -1)  # on the rl dim
        hier2_re_rel = torch.mm(hier2_rel_index, self.hier2_relation_matrix.weight.transpose(0, 1))  # bs, hs
        # gate
        hier2_concat_rel = torch.cat((x, hier2_re_rel), 1)  # bs, 2hs
        hier2_gate_rel = torch.sigmoid(self.hier2_fc1(hier2_concat_rel))
        hier2_context_rel = hier2_gate_rel * x + (1 - hier2_gate_rel) * hier2_re_rel
        # MLP linear
        hier2_output_rel = self.hier2_fc3(torch.relu(self.hier2_fc2(hier2_context_rel)))
        # add & norm
        hier2_output_rel += x
        hier2_output_rel = self.hier2_layer_norm(hier2_output_rel)

        hier3_rel_logits = torch.mm(x, self.hier3_relation_matrix.weight)  # ->(bs, rl)
        hier3_rel_index = torch.softmax(hier3_rel_logits, -1)  # on the rl dim
        hier3_re_rel = torch.mm(hier3_rel_index, self.hier3_relation_matrix.weight.transpose(0, 1))  # bs, hs
        # gate
        hier3_concat_rel = torch.cat((x, hier3_re_rel), 1)  # bs, 2hs
        hier3_gate_rel = torch.sigmoid(self.hier3_fc1(hier3_concat_rel))
        hier3_context_rel = hier3_gate_rel * x + (1 - hier3_gate_rel) * hier3_re_rel
        # MLP linear
        hier3_output_rel = self.hier3_fc3(torch.relu(self.hier3_fc2(hier3_context_rel)))
        # add & norm
        hier3_output_rel += x
        hier3_output_rel = self.hier3_layer_norm(hier3_output_rel)


        return hier1_rel_logits, hier1_output_rel, hier2_rel_logits, hier2_output_rel, hier3_rel_logits, hier3_output_rel