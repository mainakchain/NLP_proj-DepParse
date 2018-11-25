from outparse import OptionParser
import pickle, utils, mstlstm, os, os.path, time
import torch

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train",dest="conll_train",help="Annotated conll train file",metavar="FILE",
                        default="../data/train.conll.pb")
    parser.add_option("--dev",dest="conll_dev",help="Annotated conll dev file",metavar="FILE",
                        default="../data/dev.conll.pb")
    parser.add_option("--test",dest="conll_test",help="Annotated conll test file",metavar="FILE",
                        default="../data/test.conll.pb")

    parser.add_option("--extrn",dest="external_embedding", help="External Embeddings",metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="neuralfirstorder.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)

    parser.add_option("--numthread", type="int", dest="nthread", default=1)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)

    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--disablelabels", action="store_false", dest="labelsFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--bibi-lstm", action="store_true", dest="bibiFlag", default=False)
    parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)

    (options, parser) = parser.parse_args()
    torch.set_num_threads(options.nthread)
    print ("Number of threads", torch.get_num_threads())

    if  (options.predictFlag):
        with open(options.params, 'r') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = options.external_embedding

        print("Initializing LSTM mstparser...\n")
        parser = mstlstm.MSTParserLSTM(words, pos, rels, w2i, stored_ppt)
        parser.load(options.model)
        conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
        testpath = os.path.join(options.output, 'test_pred.conll' if not conllu else 'test_pred.conllu')

        ts = time.time()
        test_res = list(parser.predict(options.conll_test))
        te = time.time()

        print("Finished predicting test ", te - ts, "seconds.\n")

        if not conllu:
            os.system('perl src/utils/eval.pl -g ' + options.conll_test + ' -s ' + testpath + ' > ' + testpath + '.txt')
        else:
            os.system(
                'python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + options.conll_test + ' ' + testpath + ' > ' + testpath + '.txt')
            with open(testpath + '.txt', 'rb') as f:
                for l in f:
                    if l.startswith('UAS'):
                        print('UAS:%s' % l.strip().split()[-1])
                    elif l.startswith('LAS'):
                        print('LAS:%s' % l.strip().split()[-1])

    else:
        print("Preparing Vocab...")
        words, w2i, pos, rels = utils.vocab(options.conll_train)

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, w2i, pos, rels, options), paramsfp)
        print("Finished collecting vocab")

        print("Initializing LSTM mstparser")
        parser = mstlstm.MSTParserLSTM(words, pos, rels, w2i, options)

        for epoch in xrange(options.epochs):
            print('Starting epoch: ', epoch)
            parser.train(options.conll_train)
            conllu = (os.path.splitext(options.conll_dev.lower())[1] == '.conllu')
            devpath = os.path.join(options.output,
                                   'dev_epoch_' + str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
            utils.write_conll(devpath, parser.predict(options.conll_dev))
            parser.save(os.path.join(options.output, os.path.basename(options.model) + str(epoch + 1)))

        if not conllu:
                os.system(
                    'perl src/utils/eval.pl -g ' + options.conll_dev + ' -s ' + devpath + ' > ' + devpath + '.txt')
            else:
                os.system(
                    'python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
                with open(devpath + '.txt', 'rb') as f:
                    for l in f:
                        if l.startswith('UAS'):
                            print('UAS:%s' % l.strip().split()[-1])
                        elif l.startswith('LAS'):
                            print ('LAS:%s' % l.strip().split()[-1])
