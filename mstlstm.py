import shutil
from torch.nn import nn
from torch.nn.functional as F
from torch.nn.init import *
from torch import optim
from utils import read_conll
from operator import itemgetter
import utils, time, random, decoder
import numpy as np

import os

if 'GPU' not in os.environ or int(os.environ("GPU")) == 0:
    print("using CPU")
    use_gpu = False
else:
    print("using GPU")
    use_gpu = True

get_data = (lambda x: x.data.cpu()) if use_gpu else (lambda x: x.data)

def Variable(inner):
    """
    initializes the autograd of the variable according to using the gpu or not
    """
    return torch.autograd.Variable(inner.cuda() if use_gpu else inner)

def Parameter(shape=None, init=xavier_uniform):
    """
    initializes Tensor with the input shape (if present)
    """
    if hasattr(init, 'shape'):
        assert not shape
        return nn.Parameter(torch.Tensor(init))

    shape = (shape, 1) if type(shape) == int else shape
    return nn.Parameter(init(torch.Tensor(*shape)))

def scalar(f):
    """
    initializes the variable f according to it's type
    """
    if type(f) == int:
        return Variable(torch.LongTensor([f]))
    if type(f) == float:
        return Variable(torch.FloatTensor([f]))

def cat(l, dimension=-1):
    """
    concatenates the sequence of tensors 'valid_l' in the given 'dimension'
    """
    valid_l = filter(lambda x: x, l)
    if dimension < 0:
        dimension += len(valid_l[0].size())
    return torch.cat(valid_l, dimension)

class RNNState():
    """
    Class to define a state in RNN
    """
    def __init__(self, cell, hidden=None):
        self.cell = cell
        self.hidden = hidden
        if not hidden:
            self.hidden = Variable(torch.zeros(1, self.cell.hidden_size)), \
                          Variable(torch.zeros(1, self.cell.hidden_size))

    def next(self, input):
        return RNNState(self.cell, self.cell(input, self.hidden))

    def __call__(self):
        return self.hidden[0]

class MSTParserLSTM(nn.Module):
    """
    Tree based parser using LSTM. Inheritating torch.nn.Module class
    """
    def __init__(self, vocab, pos, rels, w2i, options):
        super(MSTParserLSTMModel, self).__init__()
        """
        Initializing the parser with the given options.
        """
        random.seed(1)
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu
                            }
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind + 3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.external_embedding, self.edim = None, 0
        if options.external_embedding is not None: ##If the external embeddings are supplied
            #use external embedding
            external_embedding_fp = open(options.external_embedding, 'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0]: [float(f) for
                f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()
            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in range(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            np_emb = np.zeros((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                np_emb[i] = self.external_embedding[word]
            self.elookup = nn.Embedding(*np_emb.shape)
            self.elookup.weight = Parameter(np_emb)
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2
            #We have the final embeddings 'extrnd' ready with '*PAD*' and '*INITIAL*' token included
            print("Load external embedding with vector dimensions: ",self.edim)

        if self.bibiFlag:
            #building the parser with biLSTM
            self.builders = [nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims),
                             nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims)]

            self.bbuilders = [nn.LSTMCell(self.ldims * 2, self.ldims),
                              nn.LSTMCell(self.ldims * 2, self.ldims)]

        elif self.layers > 0:
            #If layers given, we add the LSTM cell in the RNN network
            assert self.layers == 1, 'Not yet suppporting deep LSTM'
            self.builders = [nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims),
                            nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims)]

        else:
            # No layers found then we simply frame the network with just RNN
            self.builders = [nn.RNNCell(self.wdims + self.pdims + self.edim, self.ldims),
                         nn.RNNCell(self.wdims + self.pdims + self.edim, self.ldims)]

        for i, b in enumerate(self.builders):
            self.add_module('builder%i' % i, b)
        if hasattr(self, 'bbuilders'):
            for i, b in enumerate(self.bbuilders):
                self.add_module('bbuilder%i' % i, b)
                self.hidden_units = options.hidden_units
                self.hidden2_units = options.hidden2_units

                self.vocab['*PAD*'] = 1
                self.pos['*PAD*'] = 1

                self.vocab['*INITIAL*'] = 2
                self.pos['*INITIAL*'] = 2

                self.wlookup = nn.Embedding(len(vocab) + 3, self.wdims)
                self.plookup = nn.Embedding(len(pos) + 3, self.pdims)
                self.rlookup = nn.Embedding(len(rels), self.rdims)

                self.hidLayerFOH = Parameter((self.ldims * 2, self.hidden_units))
                self.hidLayerFOM = Parameter((self.ldims * 2, self.hidden_units))
                self.hidBias = Parameter((self.hidden_units))

                if self.hidden2_units:
                    self.hid2Layer = Parameter((self.hidden_units, self.hidden2_units))
                    self.hid2Bias = Parameter((self.hidden2_units))

                self.outLayer = Parameter(
                    (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, 1))

                if self.labelsFlag:
                    self.rhidLayerFOH = Parameter((2 * self.ldims, self.hidden_units))
                    self.rhidLayerFOM = Parameter((2 * self.ldims, self.hidden_units))
                    self.rhidBias = Parameter((self.hidden_units))

                    if self.hidden2_units:
                        self.rhid2Layer = Parameter((self.hidden_units, self.hidden2_units))
                        self.rhid2Bias = Parameter((self.hidden2_units))

                    self.routLayer = Parameter(
                        (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, len(self.irels)))
                    self.routBias = Parameter((len(self.irels)))
