import os
import sys
import pickle
import numpy as np
import argparse
import time
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim

from NYT10_data_loader import get_data_loader
from model.CoRA import CoRA
from utils import AverageMeter


def soft_pred(rel_num, sent_rel_label):
    num_cases = sent_rel_label.shape[0]
    n_rel = rel_num - 1
    p_rel = 1.0 - 0.1
    q_rel = 0.1 / n_rel
    soft_label = torch.full((num_cases, rel_num), q_rel).cuda()
    print(soft_label.shape)
    print(sent_rel_label.unsqueeze(1).shape)
    sent_rel_label = sent_rel_label.unsqueeze(1)
    soft_label = soft_label.scatter_(1, sent_rel_label, p_rel)
    return soft_label.cuda()


def train(train_loader, word_embedding_matrix, args):
    model = CoRA(args, word_embedding=word_embedding_matrix).cuda()
    bag_criterion = nn.CrossEntropyLoss()
    sent_hier1_critierion = nn.CrossEntropyLoss()
    sent_hier2_critierion = nn.CrossEntropyLoss()
    sent_hier3_critierion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5)
    model.zero_grad()
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    for epoch in range(40):
        t_start = time.time()
        model.train()
        print("\n=== Epoch %d train ===" % epoch)
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_pos_acc = AverageMeter()
        for i, data in enumerate(train_loader):
            for d in range(len(data)):
                data[d] = data[d].cuda()
            scope, sentences, entity1, entity2, pos1, pos2, piece_masks, hier1_rel, hier2_rel, hier3_rel, sen_len, bag_rel = data
            bag_pred, hier1_pred, hier2_pred, hier3_pred = model(scope, sentences, pos1, pos2, piece_masks, entity1, entity2, bag_rel)
            # soft_label
            # hier1_soft_label = soft_pred(args.hier1_num_classes, hier1_rel)
            # hier2_soft_label = soft_pred(args.hier2_num_classes, hier2_rel)
            # hier3_soft_label = soft_pred(args.hier3_num_classes, hier3_rel)
            # calculate the loss
            hier1_loss = sent_hier1_critierion(input=hier1_pred, target=hier1_rel)
            hier2_loss = sent_hier2_critierion(input=hier2_pred, target=hier2_rel)
            hier3_loss = sent_hier3_critierion(input=hier3_pred, target=hier3_rel)
            loss = bag_criterion(input=bag_pred, target=bag_rel) + hier1_loss + hier2_loss + hier3_loss
            _, pred = torch.max(torch.softmax(bag_pred, -1), -1)
            acc = (pred == bag_rel).sum().item() / bag_rel.shape[0]
            pos_total = (bag_rel != 0).sum().item()
            pos_correct = ((pred == bag_rel) & (bag_rel != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # log
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            t_end = time.time()
            time_str = t_end - t_start
            sys.stdout.write(
                '\rstep: %d | time: %.2f |loss: %f, acc: %f, pos_acc: %f' % (i + 1, time_str, avg_loss.avg, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.save({'state_dict': model.state_dict()}, args.model_dir+args.model_name+'_'+str(epoch)+'.pt')



def test(args, test_loader, word_embedding_matrix, model_path):
    model = CoRA(args, word_embedding=word_embedding_matrix)
    checkpoint = torch.load(model_path)
    own_state = model.state_dict()
    for name, param in checkpoint['state_dict'].items():
        if name not in own_state:
            continue
        # print('name: ', name)
        # print('Param: ', param.shape)
        own_state[name].copy_(param)

    model = model.cuda()
    # start test
    model.eval()
    avg_acc = AverageMeter()
    avg_pos_acc = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            for d in range(len(data)):
                data[d] = data[d].cuda()
            scope, sentences, entity1, entity2, pos1, pos2, piece_masks, hier1_rel, hier2_rel, hier3_rel, sen_len, bag_rel = data
            logits, hier1_pred, hier2_pred, hier3_pred = model(scope, sentences, pos1, pos2, piece_masks, entity1, entity2, bag_rel)
            prob = torch.softmax(logits, -1)
            label = bag_rel.argmax(-1)
            _, pred = torch.max(prob, -1)
            acc = (pred == label).sum().item() / label.shape[0]
            pos_total = (label != 0).sum().item()
            pos_correct = ((pred == label) & (label != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | acc: %f, pos_acc: %f' % (i + 1, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            y_true.append(bag_rel[:, 1:])
            y_pred.append(prob[:, 1:])
        y_true = torch.cat(y_true).reshape(-1).detach().cpu().numpy()
        y_pred = torch.cat(y_pred).reshape(-1).detach().cpu().numpy()
    # AUC score
    auc = metrics.average_precision_score(y_true, y_pred)
    print("\n[TEST] auc: {}".format(auc))
    # P@N
    order = np.argsort(-y_pred)
    p100 = (y_true[order[:100]]).mean() * 100
    p200 = (y_true[order[:200]]).mean() * 100
    p300 = (y_true[order[:300]]).mean() * 100
    print("P@100: {0:.1f}, P@200: {1:.1f}, P@300: {2:.1f}, Mean: {3:.1f}".format(p100, p200, p300,
                                                                                 (p100 + p200 + p300) / 3))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for CoRA in pytorch')
    parser.add_argument('--model_name', default='CoRA', type=str, help='the name of end-to-end model')
    parser.add_argument('--gpu', default='1', type=str, help='gpu to use')
    parser.add_argument('--data_path', default='./data/', type=str, help='path to load data')
    parser.add_argument('--model_dir', default='./outputs/ckpt/CoRA/', type=str, help='path to store or load model')
    parser.add_argument('--is_training', default=False, action='store_true', help='Bool type for training or testing')
    parser.add_argument('--batch_train_size', default=160, type=int, help='entity numbers used each training time')
    parser.add_argument('--batch_test_size', default=262, type=int, help='entity numbers used each testing time')
    parser.add_argument('--max_epoch', default=40, type=int, help='maximum of training epochs')
    parser.add_argument('--learning_rate', default=0.1, type=int, help='learning_rate')
    parser.add_argument('--weight_decay', default=0.00001, type=int, help='weight decay')
    parser.add_argument('--drop_prob', default=0.5, type=int, help='dropout rate')

    # parser.add_argument('--word_size', config)
    parser.add_argument('--hidden_size', default=230, type=int, help='dimension of hidden feature')
    parser.add_argument('--word_size', default=50, type=int, help='dimension of word embedding')
    parser.add_argument('--pos_size', default=5, type=int, help='dimension of position embedding')
    parser.add_argument('--max_length', default=120, type=int, help='maximum of number of words in one sentence')
    parser.add_argument('--hier1_num_classes', default=9, type=int, help='maximum of hier1 relations')
    parser.add_argument('--hier2_num_classes', default=36, type=int, help='maximum of hier2 relations')
    parser.add_argument('--hier3_num_classes', default=53, type=int, help='maximum of hier3 relations')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('is_training: ', args.is_training)

    word_embedding_matrix = pickle.load(open(os.path.join(args.data_path, 'initial_vectors/init_vec'), 'rb'))['wordvec']

    if args.is_training:
        train_loader = get_data_loader(args, shuffle=True, batch_size=args.batch_train_size)
        train(train_loader, word_embedding_matrix, args)
    else:
        test_loader = get_data_loader(args, shuffle=False, batch_size=args.batch_test_size)
        model_path = args.model_dir + args.model_name + '_'
        for i in range(40): # CoRA 17
        # i = 13
            print("\n === Test: {} epoch===".format(i))
            test(args, test_loader, word_embedding_matrix, model_path + str(i) + '.pt')


