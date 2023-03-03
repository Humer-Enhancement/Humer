# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import os

import torch
import gensim
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from config import Config
from model import TextCNN
# from model import BiGRU
import argparse



CLIP = 0.20
# EPOCH_LIST = [5, 5, 20]
# EPOCH_LIST = [1, 1, 1, 1, 1, 1, 1, 1, 1, 20]
# EPOCH_LIST = [3, 3, 3, 3, 20]

wrong_id_set = set()

# Data Cleansing
def text_to_wordlist(text):
    text = text.split()
    return text


# Create a Language class that will keep track of the dataset vocabulary and corresponding indices
class Language:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words + 1
            self.index2word[self.n_words + 1] = word
            self.n_words += 1


class TextDataset(Dataset):
    def __init__(self, data_list, word2index, labels, ids):
        self.data_list = data_list
        self.labels = labels
        self.word2index = word2index
        self.ids = ids

    def __getitem__(self, index):
        q = self.data_list[index]
        q_indices = []
        for word in q:
            q_indices.append(self.word2index[word])

        return q_indices, self.labels[index], self.ids[index]

    def __len__(self):
        return len(self.data_list)


class CustomCollate:
    def custom_collate(self, batch):
        # batch = list of tuples where each tuple is of the form ([data1, data2, data3], [lb1, lb2, lb3], [id1, id2, id3])
        q_list = []
        labels = []
        ids = []
        for training_example in batch:
            q_list.append(training_example[0])
            labels.append(training_example[1])
            ids.append(training_example[2])

        q_lengths = [len(q) for q in q_list]

        return q_list, q_lengths, labels, ids

    def __call__(self, batch):
        return self.custom_collate(batch)


def load_data_from_file(data_path, file_tag):
    '''
    -1 is load test_dataset or wrong_dataset

    '''

    if file_tag != -1:
        # load train dataset
        data_path = data_path + 'data_'+ str(file_tag) + '.csv'

    df_file = pd.read_csv(data_path, sep='##::##', engine='python')
    print('data_path is ' + data_path)
    print(df_file)

    res = []
    data_list = []
    data_labels = []
    data_ids = []
    for _,row in df_file.iterrows():
        q = text_to_wordlist(str(row['s']))

        label = int(row['label'])
        id = row['id']
        if q:
            data_list.append(q)
            data_labels.append(label)
            data_ids.append(id)

    res.append(data_list)
    res.append(data_labels)
    res.append(data_ids)

    return res


def course_learning_train(args, training_iter, test_loader, model, criterion, optimizer, sof, epoch_num, total_epoch_state, total_epoch):


    for epoch in range(epoch_num):

        print('\n\n')
        loss_history = []
        train_correct_total = 0

        model.train(True)

        for i, (data, data_len, label, ids) in enumerate(training_iter):
            

            label = torch.LongTensor(label).to(device)

            out = model(data, data_len)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

            optimizer.step()

            predictions = sof(out)
            total = label.size()[0]

            correct = (torch.max(predictions, 1)[1] == label).sum().item()
            train_correct_total += correct

            loss_history.append(loss.item())

            if (i + 1) % 50 == 0:
                print('Total epoch [{}/{}], Step epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(
                    total_epoch_state + 1, total_epoch,
                    epoch + 1, epoch_num,
                    config.batch_size * i, len(train_list),
                    np.mean(loss_history),
                    (correct / total) * 100))

        print('Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(np.mean(loss_history), (
        train_correct_total / len(train_list)) * 100))

        # save the model in every epoch
        save_path = args.output_dir + args.train_dir
        print('save_path is : {}'.format(save_path))

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model, save_path + args.mode + '_model_' + str(epoch + 1) + '.pkl')


        # Evaluate the model
        model.eval()

        val_correct_total = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        tmp_str = args.test_dir[1:]
        output_f = open(save_path + args.mode + 'model_' + '_epoch_' + str(epoch + 1) + 'testfile_' + tmp_str.replace('/', '-') + '.txt', 'a+')
        output_f.write('Device:  {}\n'.format(device))
        output_f.write('Test Data file: {}\n'.format(TEST_FILE_PATH))
        output_f.write('Test Data length:  {}\n'.format(len(test_list)))
        output_f.write('Total Unique Vocabulary Words:  {}\n'.format(n_vocabulary_words))
        output_f.write('*********Load the model path is: {}*********\n'.format(save_path + args.mode + 'model_' + str(epoch + 1) + '.pkl'))

        with torch.no_grad():
            for i, (data, data_len, label, ids) in enumerate(test_loader):
                
                label = torch.LongTensor(label).to(device)

                out = model(data, data_len)

                predictions = sof(out)
                
                if i % 1000 == 0:
                    print('Testing...[{}/{}]'.format(i, len(test_list)))
                
                tmp_label = int(label.item())
                tmp_pred = int(torch.max(predictions, 1)[1].item())

                output_f.write('line: {}, id: [{}], label: {}, Prob of positive: {:.6f}\n'.format(
                    i + 2, ids[0], tmp_label, predictions[0][1].item()))
                

                if tmp_label == 1 and tmp_pred == 1:
                    TP += 1
                elif tmp_label == 0 and tmp_pred == 0:
                    TN += 1
                elif tmp_label == 0 and tmp_pred == 1:
                    FP += 1
                    if total_epoch_state + 1 == total_epoch:
                        wrong_id_set.add(ids[0])
                elif tmp_label == 1 and tmp_pred == 0:
                    FN += 1
                    if total_epoch_state + 1 == total_epoch:
                        wrong_id_set.add(ids[0])
                else:
                    raise AssertionError


                correct = (torch.max(predictions, 1)[1] == label).sum().item()
                val_correct_total += correct

            avg_acc_val = val_correct_total / len(test_list)
            print('Testing Set Size {}, Correct in test_data_set {}, Accuracy {:.6f}'.format(
                    len(test_list),
                    val_correct_total,
                    avg_acc_val)) 

            acc = (TP+TN) / (TP+TN+FP+FN)
            if (TP + FP) == 0:
                prec = 0
            else:
                prec = TP/(TP+FP)
            
            if (TP + FN) == 0:
                rec = 0
            else:
                rec = TP/(TP+FN)
            
            if (prec + rec) == 0:
                f1 = 0
            else:
                f1 = 2*(prec * rec)/(prec + rec)

            print('Testing Set Size {}, TP {}, FP {}, TN {}, FN {}, Accuracy {:.6f}, Precision {:.6f}, Recall {:.6f}, F1 {:.6f}'.format(
                    len(test_list),
                    TP, FP, TN, FN,
                    acc, prec, rec, f1))

            output_f.write('Testing Set Size {}, Correct in test_data_set {}, Accuracy {:.6f}\n'.format(
                    len(test_list),
                    val_correct_total,
                    avg_acc_val))
        
            output_f.write('Test dataset is data{}.csv, the pkl path is{}.\n'.format(
                    str(args.test_dir),
                    save_path + args.mode + 'model_' + str(epoch + 1) + '.pkl'))

            output_f.write('Testing Set Size {}, TP {}, FP {}, TN {}, FN {}, Accuracy {:.6f}, Precision {:.6f}, Recall {:.6f}, F1 {:.6f}\n'.format(
                    len(test_list),
                    TP, FP, TN, FN,
                    acc, prec, rec, f1))


        output_f.close()
        model.train(True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--out_channel', type=int, default=2)
    parser.add_argument('--label_num', type=int, default=2)
    parser.add_argument('--train_dir', help='For example: split_3/, split_5/ or split_10/', required=True)
    parser.add_argument('--test_dir', help='For example: ./data/train/split_3/data_0', required=True)
    parser.add_argument('--output_dir', help='For example: ./output1/ ...', required=True)

    parser.add_argument('--mode', help='no_wrong_dataset or wrong_dataset', required=True, choices=['wrong_dataset', 'no_wrong_dataset'])
    parser.add_argument('--split_num', type=int, default=3, required=True)
    parser.add_argument('--course_learning_epoch', help='5-5-20', required=True)
    

    args = parser.parse_args()
    
    device = torch.device("cuda:{}".format(args.gpu))
    print('Device: ', device)

    # Create the configuration
    config = Config(sentence_max_size=22012,
                batch_size=args.batch_size,
                label_num=args.label_num,
                learning_rate=args.lr,
                # epoch=args.epoch,
                gpu=args.gpu,
                out_channel=args.out_channel)

    SPLIT_NUM = args.split_num
    print('Split_num is {}\n'.format(SPLIT_NUM))
    
    TRAIN_FILES_PATH = './data/train/' + args.train_dir
    TEST_FILE_PATH = args.test_dir + '.csv'
    WRONG_FILE_PATH = './data/train/data_wrong.csv'

    EMBEDDING_PATH = './data/data_all/trained_vector.vector'
    EMBEDDING_DIMENSION = 200

    print('Train Data dir is {}, there are {} files in this dir'.format(
        TRAIN_FILES_PATH, SPLIT_NUM
    ))
    print('Test Data file: ' + TEST_FILE_PATH)


    # Load Train and Test Files
    train_data_list = [] #[train_data_tuple_1, train_data_tuple_2, train_data_tuple_3, ...]

    for i in range(SPLIT_NUM):

        # load_data_from_file from 0 -> SPLIT_NUM-1 is easy -> hard
        train_data_tuple = load_data_from_file(TRAIN_FILES_PATH, i) # train_data_tuple is [train_list, train_labels, train_ids]
        print('Train Data {} length: {}'.format(i, len(train_data_tuple[0])))

        train_data_list.append(train_data_tuple)

    
    # test_data_tuple is [test_list, test_labels, test_ids]
    test_data_tuple = load_data_from_file(TEST_FILE_PATH, -1)
    print('Test Data length: ', len(test_data_tuple[0]))

    # wrong_data_tuple is [wrong_list, wrong_labels, wrong_ids]
    wrong_data_tuple = []
    if args.mode == 'wrong_dataset':
        wrong_data_tuple.extend(load_data_from_file(WRONG_FILE_PATH, -1))
        print('Wrong Data length: ', len(wrong_data_tuple[0]))



    language = Language()
    
    data_list = []
    for i in train_data_list:
        data_list.append(i[0])
    data_list.append(test_data_tuple[0])

    if args.mode == 'wrong_dataset':
        data_list.append(wrong_data_tuple[0])
        print('\n\n##########Successfully extend the wrong words into the vocabulary.##########\n')

    for data in data_list:
        for q in data:
            language.addSentence(q)

    n_vocabulary_words = len(language.word2index)
    print('Total Unique Vocabulary Words: ', n_vocabulary_words)


    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=False)
    word2vec_weights = torch.FloatTensor(word2vec_model.vectors)

    tempCount = 0

    weights = torch.randn(n_vocabulary_words + 1, EMBEDDING_DIMENSION)
    weights[0] = torch.zeros(EMBEDDING_DIMENSION)
    for word, lang_word_index in language.word2index.items():
        if word in word2vec_model:
            weights[lang_word_index] = torch.FloatTensor(word2vec_model.word_vec(word))
            tempCount += 1

    print('tempCount:', tempCount, 'n_vocabulary_words:', n_vocabulary_words)

    del word2vec_model
    del word2vec_weights


    model = TextCNN(config, weights).to(device)
    # model = BiGRU.BiGRU(config, weights).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.RMSprop(filter(lambda x: x.requires_grad, model.parameters()), lr=config.lr)
    sof = nn.Softmax(dim=1)


    # Train
    train_list = []
    train_labels = []
    train_ids = []
    test_list = test_data_tuple[0]
    test_labels = test_data_tuple[1]
    test_ids = test_data_tuple[2]

    if args.mode == 'wrong_dataset':
        train_list.extend(wrong_data_tuple[0])
        train_labels.extend(wrong_data_tuple[1])
        train_ids.extend(wrong_data_tuple[2])
        print('\n\n##########Successfully extend the wrong datas into train_list.##########\n')
    

    #begin course learning
    EPOCH_LIST = [int(kk) for kk in args.course_learning_epoch.split('-')]
    print('Epoch list is {}'.format(EPOCH_LIST))

    for i in range(SPLIT_NUM):
        train_list.extend(train_data_list[i][0])
        train_labels.extend(train_data_list[i][1])
        train_ids.extend(train_data_list[i][2])


        train_dataset = TextDataset(train_list, language.word2index, train_labels, train_ids)
        training_iter = DataLoader(dataset=train_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    collate_fn=CustomCollate())


        test_dataset = TextDataset(test_list, language.word2index, test_labels, test_ids)
        test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=CustomCollate())


        print('Course learning total EPOCH is {}, Training Set Size is {}'.format(SPLIT_NUM, len(train_list)))
        print('Course learning total EPOCH is {}, Testing Set Size is {}'.format(SPLIT_NUM, len(test_list)))

        
        course_learning_train(args, training_iter, test_loader, model, criterion, optimizer, sof, EPOCH_LIST[i], i, SPLIT_NUM)

    with open(args.output_dir + 'wrong_ids.txt', 'a+') as f:
        f.write('--------------\n')
        for i in wrong_id_set:
            f.write(str(i) + '\n')
        f.close()

