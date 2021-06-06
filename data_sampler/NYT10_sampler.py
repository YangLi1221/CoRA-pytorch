import os
import sys
import numpy as np
from random import sample

class NYT10_data_sampler(object):

    def __init__(self, args):
        self.data_path = args.data_path + "/raw_data/"
        self.processed_data_path = args.data_path + "/preprocessed_nyt10/"
        self.fixlen = 120 # length of sentence
        self.maxlen = 100 # for pos embedding
        self.is_training = args.is_training

        if self.is_training:
            self.suffix = "train"
        else:
            self.suffix = "test"

        assert os.path.exists(self.data_path)

        if os.path.exists(os.path.join(self.processed_data_path, self.suffix + '_preprocessed.npy')):
            self._init_relation()
            self.data = np.load(os.path.join(self.processed_data_path, self.suffix + '_preprocessed.npy'), allow_pickle=True)
        else:
            if not os.path.exists(self.processed_data_path):
                os.makedirs(self.processed_data_path)
            if not os.path.exists(os.path.join(self.data_path, self.suffix + '_sorted.npy')):
                self._init_relation()
                self._sort_data()
            self._init_relation()
            self._init_word()
            self._preprocessed_data()


    def _pos_embedding(self, x):
        return max(0, min(x + self.maxlen, self.maxlen + self.maxlen + 1))

    def _init_relation(self):
        self.hier1_rel2id = {}
        self.hier2_rel2id = {}
        self.hier3_rel2id = {}
        print(" Reading relation ids... ")
        i_hier1 = 0
        i_hier2 = 0
        self.hier1_rel2id['NA'] = 0
        self.hier2_rel2id['NA'] = 0
        f = open(self.data_path + "relation2id.txt", "r")
        total = (int)(f.readline().strip())
        for i in range(total):
            content = f.readline().strip().split()
            self.hier3_rel2id[content[0]] = int(content[1])
            if content[0] != 'NA':
                relation_split = content[0].strip().split('/')
                hier1_relation = relation_split[1]
                hier2_relation = "/" + relation_split[1] + "/" + relation_split[2]
                if hier1_relation not in self.hier1_rel2id:
                    i_hier1 += 1
                    self.hier1_rel2id[hier1_relation] = i_hier1
                if hier2_relation not in self.hier2_rel2id:
                    i_hier2 += 1
                    self.hier2_rel2id[hier2_relation] = i_hier2
        f.close()
        

    def _init_word(self):
        # reading word embedding data...
        self.word2id = {}
        self.word_size = 0
        self.word_vec = None
        print('reading word embedding data...')
        f = open(self.data_path + 'vec.txt', "r")
        total, size = f.readline().strip().split()[:2]
        total = (int)(total)
        word_size = (int)(size)
        vec = np.ones((total, word_size), dtype=np.float32)
        for i in range(total):
            content = f.readline().strip().split()
            self.word2id[content[0]] = len(self.word2id)
            for j in range(word_size):
                vec[i][j] = (float)(content[j + 1])
        f.close()
        self.word2id['UNK'] = len(self.word2id)
        self.word2id['BLANK'] = len(self.word2id)
        self.word_vec = vec


    def _split_relation(self, rel_name):
        if rel_name not in self.hier3_rel2id:
            rel_name = 'NA'
        if rel_name == 'NA':
            hier3_rel_id = self.hier3_rel2id['NA']
            hier3_rel = 'NA'
            hier2_rel_id = self.hier2_rel2id['NA']
            hier2_rel = 'NA'
            hier1_rel_id = self.hier1_rel2id['NA']
            hier1_rel = 'NA'
        else:
            hier3_rel_id = self.hier3_rel2id[rel_name]
            hier3_rel = rel_name
            relation_split = rel_name.strip().split('/')
            hier1_rel = relation_split[1]
            hier1_rel_id = self.hier1_rel2id[hier1_rel]
            hier2_rel = '/' + relation_split[1] + '/' + relation_split[2]
            hier2_rel_id = self.hier2_rel2id[hier2_rel]
        return hier1_rel, hier1_rel_id, hier2_rel, hier2_rel_id, hier3_rel, hier3_rel_id


    def _sort_data(self):
        if self.is_training:
            data_path = os.path.join(self.data_path, "train.txt")
        else:
            data_path = os.path.join(self.data_path, "test.txt")
        ori_data = []
        with open(data_path) as f:
            whole_corpus = f.readlines()
            for line in whole_corpus:
                if line == '':
                    break
                line = line.strip().split()
                rel_name = line[4]
                hier1_rel, hier1_rel_id, hier2_rel, hier2_rel_id, hier3_rel, hier3_rel_id = self._split_relation(rel_name)
                sample = {
                    "en1_id": line[0],
                    "en2_id": line[1],
                    "entity1": line[2],
                    "entity2": line[3],
                    "hier1_rel_id": hier1_rel_id,
                    "hier2_rel_id": hier2_rel_id,
                    "hier3_rel_id": hier3_rel_id,
                    "sentence": line[5:-1]
                }
                ori_data.append(sample)
        print(" Sort data... ")
        ori_data.sort(key=lambda a: a['en1_id'] + '#' + a['en2_id'] + '#' + str(a['hier3_rel_id']))
        print(" Finishing sorting data... ")
        print(" Saving the sorted data... ")
        np.save(os.path.join(self.data_path,  self.suffix + '_sorted.npy'), ori_data)
        print(" Saved the sorted data... ")


    def _preprocessed_data(self):
        max_pos_whole = 0
        print(" Loading sorted data... ")
        sorted_data = np.load(os.path.join(self.data_path, self.suffix + '_sorted.npy'), allow_pickle=True)
        print(" Pre-processing data... ")
        self.data = []
        if not self.is_training:
            self.data_pone = []
            self.data_ptwo = []
        bag = {
            "sentences": [],
            "entity1": [],
            "entity2": [],
            "pos1": [],
            "pos2": [],
            "piece_mask": [],
            "hier1_rel_id": [],
            "hier2_rel_id": [],
            "hier3_rel_id": [],
            "sen_len": []
        }
        former_bag = None
        for ins in sorted_data:

            # build bag
            if self.is_training:
                current_bag = (ins['en1_id'], ins['en2_id'], ins['hier3_rel_id'])
            else:
                current_bag = (ins['en1_id'], ins['en2_id'])

            if current_bag != former_bag:
                if former_bag is not None:
                    bag_length = len(bag['sentences'])
                    if self.is_training:
                        # self.data.append(bag)
                        # """
                        if bag_length < 15:
                            self.data.append(bag)
                        else:
                            iteration_bag = bag_length // 15
                            remainder_bag = bag_length % 15
                            for iter in range(iteration_bag):
                                tmp_bag = {}
                                slicing_start = iter * 15
                                slicing_end = (iter + 1) * 15
                                for key, values in bag.items():
                                    tmp_bag[key] = values[slicing_start:slicing_end]
                                self.data.append(tmp_bag)
                            if remainder_bag > 0:
                                tmp_bag = {}
                                for key, values in bag.items():
                                    tmp_bag[key] = values[iteration_bag * 15:]
                                self.data.append(tmp_bag)
                        # """
                    else:
                        self.data.append(bag)
                        idx_pone = np.random.randint(0, len(bag['sentences']), size=1)
                        if len(bag['sentences']) == 1:
                            idx_ptwo = np.random.randint(0, len(bag['sentences']), size=2)
                        else:
                            idx_ptwo = np.random.choice(np.arange(0, len(bag['sentences'])), 2, replace=False)
                        sample_pone = {}
                        sample_ptwo = {}
                        for key, values in bag.items():
                            sample_pone[key] = [values[idx_pone[0]]]
                            sample_ptwo[key] = [values[idx_ptwo[0]]]
                            sample_ptwo[key].append(values[idx_ptwo[1]])
                        self.data_pone.append(sample_pone)
                        self.data_ptwo.append(sample_ptwo)
                    bag = {
                        "sentences": [],
                        "entity1": [],
                        "entity2": [],
                        "pos1": [],
                        "pos2": [],
                        "piece_mask": [],
                        "hier1_rel_id": [],
                        "hier2_rel_id": [],
                        "hier3_rel_id": [],
                        "sen_len": []
                    }
                former_bag = current_bag

            # reltaion process
            bag['hier1_rel_id'].append(ins['hier1_rel_id'])
            bag['hier2_rel_id'].append(ins['hier2_rel_id'])
            bag['hier3_rel_id'].append(ins['hier3_rel_id'])

            # entity process
            sentence = ins['sentence']
            en1 = ins['entity1']
            en2 = ins['entity2']
            for i in range(len(sentence)):
                if sentence[i] == en1:
                    en1pos = i
                if sentence[i] == en2:
                    en2pos = i
            en_first = min(en1pos, en2pos)
            en_second = en1pos + en2pos - en_first
            en1_ids = self.word2id[en1] if en1 in self.word2id else self.word2id['UNK']
            en2_ids = self.word2id[en2] if en2 in self.word2id else self.word2id['UNK']
            bag['entity1'].append(en1_ids)
            bag['entity2'].append(en2_ids)

            # sentence, relative position and piecewise mask
            sen_word = np.zeros(shape=(self.fixlen), dtype=np.int32)
            relative_pos_1 = np.zeros(shape=(self.fixlen), dtype=np.int32)
            relative_pos_2 = np.zeros(shape=(self.fixlen), dtype=np.int32)
            piece_mask = np.zeros(shape=(self.fixlen), dtype=np.int32)
            for i in range(self.fixlen):
                sen_word[i] = self.word2id['BLANK']
                relative_pos_1[i] = self._pos_embedding(i-en1pos)
                relative_pos_2[i] = self._pos_embedding(i-en2pos)
                if i >= len(sentence):
                    piece_mask[i] = 0
                elif i - en_first <= 0:
                    piece_mask[i] = 1
                elif i - en_second <= 0:
                    piece_mask[i] = 2
                else:
                    piece_mask[i] = 3
            for i, word in enumerate(sentence):
                if i >= self.fixlen:
                    break
                elif not word in self.word2id:
                    sen_word[i] = self.word2id['UNK']
                else:
                    sen_word[i] = self.word2id[word]
            sen_len = min(self.fixlen, len(sentence))
            max_pos_1 = np.max(relative_pos_1)
            max_pos_2 = np.max(relative_pos_2)
            max_pos = max(max_pos_1, max_pos_2)
            max_pos_whole = max(max_pos_whole, max_pos)
            bag['sentences'].append(sen_word)
            bag['pos1'].append(relative_pos_1)
            bag['pos2'].append(relative_pos_2)
            bag['piece_mask'].append(piece_mask)
            bag['sen_len'].append(sen_len)

        if former_bag is not None:
            self.data.append(bag)
        print(max_pos_whole)
        print(" Finish processing... ")
        print(" Storing processed file... ")

        if self.is_training:
            np.save(os.path.join(self.processed_data_path, self.suffix + '_splited_preprocessed.npy'), self.data)
        else:
            np.save(os.path.join(self.processed_data_path, self.suffix + '_preprocessed.npy'), self.data)
            np.save(os.path.join(self.processed_data_path, self.suffix + '_preprocessed_pone.npy'), self.data_pone)
            np.save(os.path.join(self.processed_data_path, self.suffix + '_preprocessed_ptwo.npy'), self.data_ptwo)


if __name__ == "__main__":
    dataset = NYT10_data_sampler(is_training=True)

