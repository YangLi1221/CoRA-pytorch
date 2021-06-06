import torch
from torch.utils.data import Dataset, DataLoader

from data_sampler.NYT10_sampler import NYT10_data_sampler
from data_sampler.NYT_52w_sampler import NYT_52w_data_sampler


class NYT10_dataset(Dataset):

    def __init__(self, args):
        super(NYT10_dataset, self).__init__()
        self.dataset = NYT10_data_sampler(args)
        self.is_training = args.is_training

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, item):
        bag = self.dataset.data[item]
        sentences = torch.tensor(bag['sentences'], dtype=torch.long)
        entity1 = torch.tensor(bag["entity1"], dtype=torch.long)
        entity2 = torch.tensor(bag["entity2"], dtype=torch.long)
        pos1 = torch.tensor(bag["pos1"], dtype=torch.long)
        pos2 = torch.tensor(bag["pos2"], dtype=torch.long)
        piece_mask = torch.tensor(bag["piece_mask"], dtype=torch.long)
        hier1_rel_id = torch.tensor(bag["hier1_rel_id"], dtype=torch.long)
        hier2_rel_id = torch.tensor(bag["hier2_rel_id"], dtype=torch.long)
        hier3_rel_id = torch.tensor(bag["hier3_rel_id"], dtype=torch.long)
        sen_len = torch.tensor(bag["sen_len"], dtype=torch.long)
        if self.is_training:
            bag_rel = hier3_rel_id[0]
        else:
            bag_rel = torch.zeros(len(self.dataset.hier3_rel2id), dtype=torch.long)
            for i in set(hier3_rel_id):
                bag_rel[i] = 1
        return sentences, entity1, entity2, pos1, pos2, piece_mask, hier1_rel_id, hier2_rel_id, hier3_rel_id, sen_len, bag_rel

    def collate_fn(self, batch_data):
        bag_data = list(zip(*batch_data))
        sentences, entity1, entity2, pos1, pos2, piece_mask, hier1_rel_id, hier2_rel_id, hier3_rel_id, sen_len, bag_rel = bag_data
        scope = []
        ind = 0
        for bag in sentences:
            scope.append((ind, ind + len(bag)))
            ind += len(bag)
        scope = torch.tensor(scope, dtype=torch.int32)
        sentences = torch.cat(sentences, 0)
        entity1 = torch.cat(entity1, 0)
        entity2 = torch.cat(entity2, 0)
        pos1 = torch.cat(pos1, 0)
        pos2 = torch.cat(pos2, 0)
        piece_mask = torch.cat(piece_mask, 0)
        hier1_rel_id = torch.cat(hier1_rel_id, 0)
        hier2_rel_id = torch.cat(hier2_rel_id, 0)
        hier3_rel_id = torch.cat(hier3_rel_id, 0)
        sen_len = torch.cat(sen_len, 0)
        bag_rel = torch.stack(bag_rel)
        return scope, sentences, entity1, entity2, pos1, pos2, piece_mask, hier1_rel_id, hier2_rel_id, hier3_rel_id, sen_len, bag_rel


def get_data_loader(args, shuffle=True, batch_size=160):
    dataset = NYT10_dataset(args)
    print('Data length: ', len(dataset.dataset.data))
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=1,
        collate_fn=dataset.collate_fn)
    return data_loader