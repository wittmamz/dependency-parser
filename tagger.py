import codecs
import time
import cPickle
import gzip
import random
import os

import modules.token as tk
import modules.arc as ar

from modules.cle import CLE
from modules.evaluation import evaluate
#from modules.affixes import find_affixes


class depParser(object):
    def __init__(self):

        pass

    # save the model (weight vectors) to a file:
    def save(self, file_name, model):
        stream = gzip.open(file_name, "wb")
        cPickle.dump(model, stream)
        stream.close()

    # load the model (weight vectors) from a file:

    def load(self, file_name):
        stream = gzip.open(file_name, "rb")
        model = cPickle.load(stream)
        stream.close()
        return model

    def getTreeVec(self, arcs):
        result = {}
        for arc in arcs:
            for feature in arc.sparse_feat_vec:
                if feature in result:
                    result[feature] += 1
                else:
                    result[feature] = 1
        return result

    def getGraph(self, sentence):
        result = [set([]), sentence]
        for arc in sentence:
            result[0].add(arc.head)
            result[0].add(arc.dependent)
        result[0] = sorted(list(result[0]), key = lambda x : x.t_id)
        return result


    # train the classifiers using the perceptron algorithm:
    def train(self, file_in, file_out, max_iterations, decrease_alpha, shuffle_sentences, batch_training, sentence_limit):
    
        print "\tTraining file: " + file_in

        print "\tExtracting features"
        x0 = time.time()
        feat_vec = self.extractFeatures(file_in, sentence_limit)
        x1 = time.time()
        print "\t" + str(len(feat_vec)) + " features extracted"
        print "\t\t" + str(x1 - x0) + " sec."

        print "\tCreating arcs with feature vectors"
        y0 = time.time()
        sentences = []  # save all instantiated tokens from training data, with finished feature vectors

        # read in sentences from file and generates the corresponding token objects:
        counter = 0
        for sentence in self.sentences(codecs.open(file_in, encoding='utf-8')):
            if sentence_limit != -1 and counter == sentence_limit:
                break
            arcs = []
            # create sparse feature vector representation for each token:
            for arc in sentence:
                arc.createFeatureVector(feat_vec)
                arcs.append(arc)
            sentences.append(arcs)
            counter += 1

        y1 = time.time()
        print "\t\t" + str(y1 - y0) + " sec."
        
        weights = [0.0 for ind in range(len(feat_vec))]

        alpha = 0.5  # smoothes the effect of adjustments

        # number of decreases of alpha during training
        # works only only exactly if max_iterations is divisible by alpha_decreases
        alpha_decreases = 5
    
        z0 = time.time()
        for i in range(1, max_iterations + 1):
            amount_correct = 0
            
            
            # batch training:
            if batch_training:
                weights_copy = [x for x in weights]
                
            print "\t\tEpoch " + str(i) + ", alpha = " + str(alpha)
            for ind, s in enumerate(sentences):
    
                # expand sparse token feature vectors into all dimensions:
                # expanded_feat_vec = t.expandFeatVec(len(feat_vec))

                max_tree = CLE(self.getGraph(s), weights)[1]
                gold_tree = [x for x in s if x.relation != None]
                
                gold_tree_vec = self.getTreeVec(gold_tree)
                max_tree_vec = self.getTreeVec(max_tree)

                max_tree_vec_keys = set(max_tree_vec.keys())
                max_tree_vec_values = set(max_tree_vec.values())
                gold_tree_vec_keys = set(gold_tree_vec.keys())
                gold_tree_vec_values = set(gold_tree_vec.values())
                
                if not (max_tree_vec_keys == gold_tree_vec_keys and max_tree_vec_values == gold_tree_vec_values):
                    if batch_training:
                        for ind2 in gold_tree_vec_keys.union(max_tree_vec_keys):
                            if ind2 in max_tree_vec and ind2 in gold_tree_vec:
                                weights_copy[ind2] = weights_copy[ind2] + alpha*(gold_tree_vec[ind2]-max_tree_vec[ind2])
                            elif ind2 in gold_tree_vec:
                                weights_copy[ind2] = weights_copy[ind2] + alpha*gold_tree_vec[ind2]
                            elif ind2 in max_tree_vec:
                                weights_copy[ind2] = weights_copy[ind2] - alpha*max_tree_vec[ind2]
                    else:
                        for ind2 in gold_tree_vec_keys.union(max_tree_vec_keys):
                            if ind2 in max_tree_vec and ind2 in gold_tree_vec:
                                weights[ind2] = weights[ind2] + alpha*(gold_tree_vec[ind2]-max_tree_vec[ind2])
                            elif ind2 in gold_tree_vec:
                                weights[ind2] = weights[ind2] + alpha*gold_tree_vec[ind2]
                            elif ind2 in max_tree_vec:
                                weights[ind2] = weights[ind2] - alpha*max_tree_vec[ind2]
                else:
                    amount_correct += 1
                    
                if (ind+1) % (len(sentences) / 10) == 0:
                        print "\t\t\t" + str(ind+1) + "/" + str(len(sentences)) + " (" + str(amount_correct) + " correct)"
                
            # apply batch results to weight vectors:
            if batch_training:
                weights = [x for x in weights_copy]

            # decrease alpha
            if decrease_alpha:
                if i % int(round(max_iterations ** 1.0 / float(alpha_decreases))) == 0:
                    # int(round(max_iterations ** 1/alpha_decreases)) is the number x, for which
                    # i % x == 0 is True exactly alpha_decreases times

                    alpha /= 2
            
            # shuffle sentences
            if shuffle_sentences:
                random.shuffle(sentences)
        
        # after training is completed, save classifier vectors (model) to file:
        self.save(file_out, [feat_vec, weights])

        z1 = time.time()
        print "\t\t" + str(z1 - z0) + " sec."

    # apply the classifiers to test data:
    def test(self, file_in, mod, file_out):

        # load classifier vectors (model) and feature vector from file:

        print "\tLoading the model and the features"
        x0 = time.time()

        model_list = self.load(mod)
        feat_vec = model_list[0]
        weights = model_list[1]

        x1 = time.time()
        print "\t" + str(len(feat_vec)) + " features loaded"
        print "\t\t" + str(x1 - x0) + " sec."

        print "\tTest file: " + file_in

        print "\tCreating tokens with feature vectors"
        y0 = time.time()
        sentences = []  # save all instantiated tokens from training data, with finished feature vectors
        empty_feat_vec_count = 0
        
        # read in sentences from file and generates the corresponding token objects:
        for sentence in self.sentences(codecs.open(file_in, encoding='utf-8'), False):
            arcs = []
            # create sparse feature vector representation for each token:
            for arc in sentence:
                arc.createFeatureVector(feat_vec)
                arcs.append(arc)
                if len(arc.sparse_feat_vec) == 0:
                    empty_feat_vec_count += 1
            sentences.append(arcs)

        print "\t\t" + str(empty_feat_vec_count) + " tokens have no features of the feature set"
        y1 = time.time()
        print "\t\t" + str(y1 - y0) + " sec."

        print "\tClassifying tokens"
        z0 = time.time()
        output = open(file_out, "w")  # temporarily save classification to file for evaluation
        
        for ind, s in enumerate(sentences):
            if ind % (len(sentences) / 10) == 0 and not ind == 0:
                    print "\t\t\t" + str(ind) + "/" + str(len(sentences))

            # expand sparse token feature vectors into all dimensions:
            # expanded_feat_vec = t.expandFeatVec(len(feat_vec))

            result = CLE(self.getGraph(s), weights)

            for token in result[0]:
                for arc in result[1]:
                    if arc.dependent.t_id == token.t_id:
                        print >> output, str(token.t_id) + "\t" + token.form.encode("utf-8") + "\t" + token.lemma.encode("utf-8") + "\t" + token.pos.encode("utf-8") + "\t_\t_\t" + str(arc.head.t_id) + "\t_\t_\t_"
                        break
            print >> output, ""
            
        output.close()

        z1 = time.time()
        print "\t\t" + str(z1 - z0) + " sec."


    # build mapping of features to vector dimensions (key=feature, value=dimension index):
    def extractFeatures(self, file_in, sentence_limit):

        feat_vec = {}

        #affixes = find_affixes(file_in, 5)

        # uppercase
        #feat_vec["uppercase"] = len(feat_vec)

        # capitalized
        #feat_vec["capitalized"] = len(feat_vec)

        #for l in affixes:
            #for affix_length in l:
                #for affix in l[affix_length]:
                    #if sum(l[affix_length][affix].values()) > 0:
                        #if affixes.index(l) == 0:
                            #feat_vec["suffix_" + affix] = len(feat_vec)
                        #elif affixes.index(l) == 1:
                            #feat_vec["prefix_" + affix] = len(feat_vec)
                        #else:
                            #feat_vec["lettercombs_" + affix] = len(feat_vec)

        # iterate over all tokens to extract features:
        counter = 0
        for sentence in self.sentences(codecs.open(file_in, encoding='utf-8')):
            if sentence_limit != -1 and counter == sentence_limit:
                break
            for arc in sentence:
            
                if not "hform_" + arc.head.form + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hform_" + arc.head.form + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "hpos_" + arc.head.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hpos_" + arc.head.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "hform_" + arc.head.form + "_hpos_" + arc.head.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hform_" + arc.head.form + "_hpos_" + arc.head.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                    
                if not "dform_" + arc.dependent.form + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["dform_" + arc.dependent.form + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "dform_" + arc.dependent.form + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["dform_" + arc.dependent.form + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                    
                if not "hform_" + arc.head.form + "_dform_" + arc.dependent.form + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hform_" + arc.head.form + "_dform_" + arc.dependent.form + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                    
                if not "hform_" + arc.head.form + "_hpos_" + arc.head.pos + "_dform_" + arc.dependent.form + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hform_" + arc.head.form + "_hpos_" + arc.head.pos + "_dform_" + arc.dependent.form + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "hform_" + arc.head.form + "_hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hform_" + arc.head.form + "_hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)    
                    
                if not "hform_" + arc.head.form + "_dform_" + arc.dependent.form + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hform_" + arc.head.form + "_dform_" + arc.dependent.form + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "hpos_" + arc.head.pos + "_dform_" + arc.dependent.form + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hpos_" + arc.head.pos + "_dform_" + arc.dependent.form + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                
                if not "hform_" + arc.head.form + "_hpos_" + arc.head.pos + "_dform_" + arc.dependent.form + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hform_" + arc.head.form + "_hpos_" + arc.head.pos + "_dform_" + arc.dependent.form + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                    
                between_pos = []
                if arc.direction == "R":
                    temp = arc.head.next_token
                    while temp.t_id != arc.dependent.t_id:
                        between_pos.append(temp.pos)
                        temp = temp.next_token
                else:
                    temp = arc.head.prev_token
                    while temp.t_id != arc.dependent.t_id:
                        between_pos.append(temp.pos)
                        temp = temp.prev_token
                if len(between_pos) == 0:
                    between_pos.append("$none$")
                
                if not "hpos_" + arc.head.pos + "_bpos_" + "_bpos_".join(between_pos) + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hpos_" + arc.head.pos + "_bpos_" + "_bpos_".join(between_pos) + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                
                if arc.head.next_token != None and arc.dependent.next_token != None:
                    if not "hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_hpos+1_" + arc.head.next_token.pos + "_dpos+1_" + arc.dependent.next_token.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                        feat_vec["hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_hpos+1_" + arc.head.next_token.pos + "_dpos+1_" + arc.dependent.next_token.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if arc.head.next_token != None and arc.dependent.prev_token != None:
                    if not "hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_hpos+1_" + arc.head.next_token.pos + "_dpos-1_" + arc.dependent.prev_token.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                        feat_vec["hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_hpos+1_" + arc.head.next_token.pos + "_dpos-1_" + arc.dependent.prev_token.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if arc.head.prev_token != None and arc.dependent.next_token != None:
                    if not "hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_hpos-1_" + arc.head.prev_token.pos + "_dpos+1_" + arc.dependent.next_token.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                        feat_vec["hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_hpos-1_" + arc.head.prev_token.pos + "_dpos+1_" + arc.dependent.next_token.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if arc.head.prev_token != None and arc.dependent.prev_token != None:
                    if not "hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_hpos-1_" + arc.head.prev_token.pos + "_dpos-1_" + arc.dependent.prev_token.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                        feat_vec["hpos_" + arc.head.pos + "_dpos_" + arc.dependent.pos + "_hpos-1_" + arc.head.prev_token.pos + "_dpos-1_" + arc.dependent.prev_token.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                
                # form length
                #if not "current_word_len_" + str(len(token.form)) in feat_vec:
                    #feat_vec["current_word_len_" + str(len(token.form))] = len(feat_vec)
                #if tid < len(sentence)-1:
                    #if not "prev_word_len_" + str(len(token.form)) in feat_vec:
                        #feat_vec["prev_word_len_" + str(len(token.form))] = len(feat_vec)
                #if tid != 0:
                    #if not "next_word_len_" + str(len(token.form)) in feat_vec:
                        #feat_vec["next_word_len_" + str(len(token.form))] = len(feat_vec)

                # position in sentence
                #if not "position_in_sentence_" + str(tid) in feat_vec:
                    #feat_vec["position_in_sentence_" + str(tid)] = len(feat_vec)
            
            counter += 1
        return feat_vec


    # a generator to read a file sentence-wise and generate an Arc object for each token:
    def sentences(self, file_stream, train=True):
        # a list of Token objects of every sentence is yielded:
        counter = 0
        counter2 = 0
        sentence = []
        if train:
            temp = [[tk.Token("ROOT", counter), -1, "_"]]
        else:
            temp = [tk.Token("ROOT", counter)]
        for line in file_stream:
            line = line.rstrip()
            if line:
                entries = line.split("\t")
                if train:
                    temp.append([tk.Token(entries, counter),int(entries[6]), entries[7]])
                else:
                    temp.append(tk.Token(entries, counter))
            elif temp:
                if train:
                    temp[-1][0].setAdjacentTokens(temp[-2][0], None)
                    temp[0][0].setAdjacentTokens(None, temp[1][0])
                    for ind in range(len(temp)-1):
                        if ind > 0:
                            temp[ind][0].setAdjacentTokens(temp[ind-1][0], temp[ind+1][0])
                        for ind2 in range(ind+1,len(temp)):
                            if ind == temp[ind2][1]:
                                sentence.append(ar.Arc(counter, counter2, temp[ind][0], temp[ind2][0], temp[ind2][2]))
                                counter2 += 1
                            else:
                                sentence.append(ar.Arc(counter, counter2, temp[ind][0], temp[ind2][0], None))
                                counter2 += 1
                            if ind > 0:
                                if ind2 == temp[ind][1]:
                                    sentence.append(ar.Arc(counter, counter2, temp[ind2][0], temp[ind][0], temp[ind][2]))
                                    counter2 += 1
                                else:
                                    sentence.append(ar.Arc(counter, counter2, temp[ind2][0], temp[ind][0], None))
                                    counter2 += 1
                else:
                    temp[-1].setAdjacentTokens(temp[-2], None)
                    temp[0].setAdjacentTokens(None, temp[1])
                    for ind in range(len(temp)-1):
                        if ind > 0:
                            temp[ind].setAdjacentTokens(temp[ind-1], temp[ind+1])
                        for ind2 in range(ind+1,len(temp)):
                            sentence.append(ar.Arc(counter, counter2, temp[ind], temp[ind2], None))
                            counter2 += 1
                            if ind > 0:
                                sentence.append(ar.Arc(counter, counter2, temp[ind2], temp[ind], None))
                                counter2 += 1
                yield sentence
                counter += 1
                counter2 = 0
                sentence = []
                if train:
                    temp = [[tk.Token("ROOT", counter), -1, "_"]]
                else:
                    temp = [tk.Token("ROOT", counter)]
        if sentence:
            yield sentence
        
if __name__ == '__main__':

    t0 = time.time()

    import argparse

    argpar = argparse.ArgumentParser(description='')

    mode = argpar.add_mutually_exclusive_group(required=True)
    mode.add_argument('-train', dest='train', action='store_true', help='run in training mode')
    mode.add_argument('-test', dest='test', action='store_true', help='run in test mode')
    mode.add_argument('-ev', dest='evaluate', action='store_true', help='run in evaluation mode')
    #mode.add_argument('-tag', dest='tag', action='store_true', help='run in tagging mode')

    argpar.add_argument('-i', '--infile', dest='in_file', help='in file', required=True)
    argpar.add_argument('-sentence-limit', dest='sentence_limit', help='sentence limit', default='-1')
    argpar.add_argument('-e', '--epochs', dest='epochs', help='epochs', default='1')
    argpar.add_argument('-m', '--model', dest='model', help='model', default='model')
    argpar.add_argument('-o', '--output', dest='output_file', help='output file', default='output.txt')
    #argpar.add_argument('-t1', '--topxform', dest='top_x_form', help='top x form', default=None)
    #argpar.add_argument('-t2', '--topxwordlen', dest='top_x_word_len', help='top x word len', default=None)
    #argpar.add_argument('-t3', '--topxposition', dest='top_x_position', help='top x position', default=None)
    #argpar.add_argument('-t4', '--topxprefix', dest='top_x_prefix', help='top x prefix', default=None)
    #argpar.add_argument('-t5', '--topxsuffix', dest='top_x_suffix', help='top x suffix', default=None)
    #argpar.add_argument('-t6', '--topxlettercombs', dest='top_x_lettercombs', help='top x letter combs', default=None)
    argpar.add_argument('-decrease-alpha', dest='decrease_alpha', action='store_true', help='decrease alpha', default=False)
    argpar.add_argument('-shuffle-sentences', dest='shuffle_sentences', action='store_true', help='shuffle sentences', default=False)
    argpar.add_argument('-batch-training', dest='batch_training', action='store_true', help='batch training', default=False)

    args = argpar.parse_args()

    d = depParser()
    if os.stat(args.in_file).st_size == 0:
        print "Input file is empty"
    else:
        if args.train:
            print "Running in training mode\n"
            #if not args.top_x_form:
                #print args.top_x_form
            #top_x = [args.top_x_form, args.top_x_word_len, args.top_x_position, args.top_x_prefix, args.top_x_suffix, args.top_x_lettercombs]
            d.train(args.in_file, args.model, int(args.epochs), args.decrease_alpha, args.shuffle_sentences, args.batch_training, int(args.sentence_limit))

        elif args.test:
            print "Running in test mode\n"
            d.test(args.in_file, args.model, args.output_file)
        elif args.evaluate:
            print "Running in evaluation mode\n"
            out_stream = open(args.output_file, 'w')
            evaluate(args.in_file, out_stream)
            out_stream.close()
    t1 = time.time()
    print "\n\tDone. Total time: " + str(t1 - t0) + "sec.\n"
