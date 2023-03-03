# —*- coding: utf-8 -*-


class Config(object):
    def __init__(
            self, 
            word_embedding_dimension=200,
            epoch=20, 
            sentence_max_size=21910, 
            cuda=True,
            gpu=3,
            label_num=2, 
            learning_rate=0.0001, 
            batch_size=32,
            out_channel=100
        ):

        self.word_embedding_dimension = word_embedding_dimension # word embedding size
        self.epoch = epoch
        self.sentence_max_size = sentence_max_size # sentence max length
        self.label_num = label_num # class number
        self.lr = learning_rate
        self.batch_size = batch_size
        self.out_channel=out_channel
        self.cuda = cuda
        self.gpu = gpu
