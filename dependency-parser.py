import codecs
import time
import cPickle
import gzip
import random
import os

import modules.token as token_class
import modules.arc as arc_class

from modules.cle import CLE
from modules.evaluation import evaluate

class depParser(object):

    def __init__(self):
    
        pass

    # save the model (weight and feature vectors) to a file:
    def save(self, file_name, model):
    
        stream = gzip.open(file_name, "wb")
        cPickle.dump(model, stream)
        stream.close()

    # load the model (weight and feature vectors) from a file:
    def load(self, file_name):
    
        stream = gzip.open(file_name, "rb")
        model = cPickle.load(stream)
        stream.close()
        return model

    # join all features for a list of arcs together in a dictionary, including how often they are represented:
    def getTreeVec(self, arcs):
    
        result = {}
        for arc in arcs:
            for feature in arc.sparse_feat_vec:
                if feature in result:
                    result[feature] += 1
                else:
                    result[feature] = 1
        return result

    # transform a list of edges (arcs) into a graph (G = {V, E}):
    def getGraph(self, arcs):
    
        result = [set([]), arcs]
        for arc in arcs:
            result[0].add(arc.head)
            result[0].add(arc.dependent)
        result[0] = sorted(list(result[0]), key = lambda x : x.t_id)
        return result


    # train the weights vector using the perceptron algorithm:
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
        sentences = []  # save all instantiated arcs from training data, with finished feature vectors

        # read in sentences from file and generate the corresponding lists of arc objects:
        sentence_counter = 0
        for sentence in self.sentences(codecs.open(file_in, encoding='utf-8')):
            # stop after the sentence limit has been reached (if applicable):
            if sentence_limit != -1 and sentence_counter == sentence_limit:
                break
                
            arcs = []
            # create sparse feature vector representation for each arc:
            for arc in sentence:
                arc.createFeatureVector(feat_vec)
                arcs.append(arc)
                
            sentences.append(arcs)
            sentence_counter += 1

        y1 = time.time()
        print "\t\t" + str(y1 - y0) + " sec."
        
        # initialize the weights vector:
        weights = [0.0 for ind in range(len(feat_vec))]

        alpha = 0.5  # smoothes the effect of adjustments

        # number of decreases of alpha during training:
        alpha_decreases = 5
    
        z0 = time.time()
        for ind in range(1, max_iterations + 1):
            amount_correct = 0          
            
            # batch training:
            if batch_training:
                weights_copy = [x for x in weights]
                
            print "\t\tEpoch " + str(ind) + ", alpha = " + str(alpha)
            for ind2, arcs in enumerate(sentences):

                gold_tree = [x for x in arcs if x.relation != None] # all arcs in a sentence with a relation label are gold arcs
                
                # get the combined feature vectors for the gold standard spanning trees:
                gold_tree_vec = self.getTreeVec(gold_tree)
                gold_tree_vec_keys = set(gold_tree_vec.keys())
                gold_tree_vec_values = set(gold_tree_vec.values())
                
                max_tree = CLE(self.getGraph(arcs), weights)[1]    

                # get the combined feature vectors for the maximum spanning trees:
                max_tree_vec = self.getTreeVec(max_tree)
                max_tree_vec_keys = set(max_tree_vec.keys())
                max_tree_vec_values = set(max_tree_vec.values())
                
                # incorrect maximum spanning tree:
                if not (max_tree_vec_keys == gold_tree_vec_keys and max_tree_vec_values == gold_tree_vec_values):
                
                    # update weights:
                    if batch_training:
                        for ind3 in gold_tree_vec_keys.union(max_tree_vec_keys):
                            if ind3 in max_tree_vec and ind3 in gold_tree_vec:
                                weights_copy[ind3] = weights_copy[ind3] + alpha * (gold_tree_vec[ind3] - max_tree_vec[ind3])
                            elif ind3 in gold_tree_vec:
                                weights_copy[ind3] = weights_copy[ind3] + alpha * gold_tree_vec[ind3]
                            elif ind3 in max_tree_vec:
                                weights_copy[ind3] = weights_copy[ind3] - alpha * max_tree_vec[ind3]
                    else:
                        for ind3 in gold_tree_vec_keys.union(max_tree_vec_keys):
                            if ind3 in max_tree_vec and ind3 in gold_tree_vec:
                                weights[ind3] = weights[ind3] + alpha * (gold_tree_vec[ind3] - max_tree_vec[ind3])
                            elif ind3 in gold_tree_vec:
                                weights[ind3] = weights[ind3] + alpha * gold_tree_vec[ind3]
                            elif ind3 in max_tree_vec:
                                weights[ind3] = weights[ind3] - alpha * max_tree_vec[ind3]
                
                # correct spanning tree:
                else:
                    amount_correct += 1
                    
                # progress output:
                if (ind2 + 1) % (len(sentences) / 10) == 0:
                        print "\t\t\t" + str(ind2 + 1) + "/" + str(len(sentences)) + " (" + str(amount_correct) + " correct)"
                
            # apply batch results to weight vector:
            if batch_training:
                weights = [x for x in weights_copy]

            # decrease alpha:
            if decrease_alpha:
                if ind % int(round(max_iterations ** 1.0 / float(alpha_decreases))) == 0:
                    # int(round(max_iterations ** 1/alpha_decreases)) is the number x, for which
                    # i % x == 0 is True exactly alpha_decreases times
                    alpha /= 2
            
            # shuffle sentences:
            if shuffle_sentences:
                random.shuffle(sentences)
        
        # after training is completed, save the model (weights and feature vectors) to file:
        self.save(file_out, [feat_vec, weights])

        z1 = time.time()
        print "\t\t" + str(z1 - z0) + " sec."

    # apply the parser to test data:
    def test(self, file_in, mod, file_out):

        # load weight and feature vectors (model) from file:
        print "\tLoading the model and the features"
        x0 = time.time()
        model_list = self.load(mod)
        feat_vec = model_list[0]
        weights = model_list[1]
        x1 = time.time()
        print "\t" + str(len(feat_vec)) + " features loaded"
        print "\t\t" + str(x1 - x0) + " sec."

        print "\tTest file: " + file_in

        print "\tCreating arcs with feature vectors"
        y0 = time.time()
        sentences = []  # save all instantiated arcs from training data, with finished feature vectors
        empty_feat_vec_count = 0
        
        # read in sentences from file and generate the corresponding lists of arc objects:
        for sentence in self.sentences(codecs.open(file_in, encoding='utf-8'), False):
            arcs = []
            
            # create sparse feature vector representation for each arc:
            for arc in sentence:
                arc.createFeatureVector(feat_vec)
                arcs.append(arc)
                if len(arc.sparse_feat_vec) == 0: # count arcs with none of the features in the feature map
                    empty_feat_vec_count += 1
            sentences.append(arcs)

        print "\t\t" + str(empty_feat_vec_count) + " arcs have no features of the feature set"
        y1 = time.time()
        print "\t\t" + str(y1 - y0) + " sec."

        print "\tParsing sentences"
        z0 = time.time()
        output = open(file_out, "w")  # save results to file for evaluation
        
        # parse sentences:
        for ind, arcs in enumerate(sentences):

            result = CLE(self.getGraph(arcs), weights)

            # output to file:
            for token in result[0]:
                for arc in result[1]:
                    if arc.dependent.t_id == token.t_id:
                        print >> output, str(token.t_id) + "\t" + token.form.encode("utf-8") + "\t" + token.lemma.encode("utf-8") + "\t" + \
                                         token.pos.encode("utf-8") + "\t_\t_\t" + str(arc.head.t_id) + "\t_\t_\t_"
                        break
            print >> output, ""
            
            # progress output:
            if (ind + 1) % (len(sentences) / 10) == 0:
                print "\t\t\t" + str(ind + 1) + "/" + str(len(sentences))
            
        output.close()

        z1 = time.time()
        print "\t\t" + str(z1 - z0) + " sec."

    # build mapping of features to vector dimensions (key=feature, value=dimension index):
    def extractFeatures(self, file_in, sentence_limit):

        feat_vec = {}

        # iterate over all arcs to extract features:
        sentence_counter = 0
        for sentence in self.sentences(codecs.open(file_in, encoding='utf-8')):
            # stop after the sentence limit has been reached (if applicable):
            if sentence_limit != -1 and sentence_counter == sentence_limit:
                break
                
            for arc in sentence:
                # unary head features:
                if not "hform_" + arc.head.form + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hform_" + arc.head.form + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "hpos_" + arc.head.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hpos_" + arc.head.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "hform_" + arc.head.form + "_hpos_" + arc.head.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["hform_" + arc.head.form + "_hpos_" + arc.head.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                    
                # unary dependent features:
                if not "dform_" + arc.dependent.form + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["dform_" + arc.dependent.form + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                if not "dform_" + arc.dependent.form + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction in feat_vec:
                    feat_vec["dform_" + arc.dependent.form + "_dpos_" + arc.dependent.pos + "_len_" + str(arc.length) + "_dir_" + arc.direction] = len(feat_vec)
                   
                # binary features (head and dependent):
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
                
                # feature taking into account the POS of all tokens between head and dependent:
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
                
                # features taking into account the token before/after the head/dependent:
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
            
            sentence_counter += 1
            
        return feat_vec


    # a generator to read a file sentence-wise and generate an arc object for each pair of tokens:
    def sentences(self, file_stream, train = True):
    
        sentence_counter = 0
        arc_counter = 0
        sentence = [] # this list is filled with arc objects and then yielded after each sentence
        tokens = [] # during processing of a sentence, this list is filled with token objects
        
        if train:
            tokens = [[token_class.Token("ROOT", sentence_counter), -1, "_"]] # create a special ROOT token for each sentence
        else: # testing mode
            tokens = [token_class.Token("ROOT", sentence_counter)] # create a special ROOT token for each sentence
            
        for line in file_stream:
            line = line.rstrip()
            if line: # next token
                entries = line.split("\t")
                if train:
                    tokens.append([token_class.Token(entries, sentence_counter),int(entries[6]), entries[7]])
                else: # testing mode
                    tokens.append(token_class.Token(entries, sentence_counter))
            elif len(tokens) > 1: # end of sentence
                if train:
                    tokens[-1][0].setAdjacentTokens(tokens[-2][0], None) # set the previous token for the last token in the sentence
                    tokens[0][0].setAdjacentTokens(None, tokens[1][0]) # set the next token for the first token in the sentence
                    for ind in range(len(tokens) - 1):
                        if ind > 0:
                            tokens[ind][0].setAdjacentTokens(tokens[ind - 1][0], tokens[ind + 1][0]) # set the previous and next tokens                    
                        # for each pair of tokens, create the corresponding arc objects:
                        for ind2 in range(ind + 1, len(tokens)):
                            if ind == tokens[ind2][1]: # gold arc
                                sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind][0], tokens[ind2][0], tokens[ind2][2]))
                                arc_counter += 1
                            else:
                                sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind][0], tokens[ind2][0], None))
                                arc_counter += 1
                            if ind > 0: # create reverse arc
                                if ind2 == tokens[ind][1]: # gold arc
                                    sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind2][0], tokens[ind][0], tokens[ind][2]))
                                    arc_counter += 1
                                else:
                                    sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind2][0], tokens[ind][0], None))
                                    arc_counter += 1
                
                else: # testing mode
                    tokens[-1].setAdjacentTokens(tokens[-2], None) # set the previous token for the last token in the sentence
                    tokens[0].setAdjacentTokens(None, tokens[1]) # set the next token for the first token in the sentence
                    for ind in range(len(tokens) - 1):
                        if ind > 0:
                            tokens[ind].setAdjacentTokens(tokens[ind-1], tokens[ind+1]) # set the previous and next tokens
                        # for each pair of tokens, create the corresponding arc objects:
                        for ind2 in range(ind + 1, len(tokens)):
                            sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind], tokens[ind2], None))
                            arc_counter += 1
                            if ind > 0:
                                sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind2], tokens[ind], None))
                                arc_counter += 1
                
                yield sentence
                sentence_counter += 1
                arc_counter = 0
                sentence = []
                
                if train:
                    tokens = [[token_class.Token("ROOT", sentence_counter), -1, "_"]] # create a special ROOT token for each sentence
                else: # testing mode
                    tokens = [token_class.Token("ROOT", sentence_counter)] # create a special ROOT token for each sentence
                    
        if len(tokens) > 1: # last sentence (not followed by an empty line)
            if train:
                tokens[-1][0].setAdjacentTokens(tokens[-2][0], None) # set the previous token for the last token in the sentence
                tokens[0][0].setAdjacentTokens(None, tokens[1][0]) # set the next token for the first token in the sentence
                for ind in range(len(tokens) - 1):
                    if ind > 0:
                        tokens[ind][0].setAdjacentTokens(tokens[ind - 1][0], tokens[ind + 1][0]) # set the previous and next tokens                    
                    # for each pair of tokens, create the corresponding arc objects:
                    for ind2 in range(ind + 1, len(tokens)):
                        if ind == tokens[ind2][1]: # gold arc
                            sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind][0], tokens[ind2][0], tokens[ind2][2]))
                            arc_counter += 1
                        else:
                            sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind][0], tokens[ind2][0], None))
                            arc_counter += 1
                        if ind > 0: # create reverse arc
                            if ind2 == tokens[ind][1]: # gold arc
                                sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind2][0], tokens[ind][0], tokens[ind][2]))
                                arc_counter += 1
                            else:
                                sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind2][0], tokens[ind][0], None))
                                arc_counter += 1
            
            else: # testing mode
                tokens[-1].setAdjacentTokens(tokens[-2], None) # set the previous token for the last token in the sentence
                tokens[0].setAdjacentTokens(None, tokens[1]) # set the next token for the first token in the sentence
                for ind in range(len(tokens) - 1):
                    if ind > 0:
                        tokens[ind].setAdjacentTokens(tokens[ind-1], tokens[ind+1]) # set the previous and next tokens
                    # for each pair of tokens, create the corresponding arc objects:
                    for ind2 in range(ind + 1, len(tokens)):
                        sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind], tokens[ind2], None))
                        arc_counter += 1
                        if ind > 0:
                            sentence.append(arc_class.Arc(sentence_counter, arc_counter, tokens[ind2], tokens[ind], None))
                            arc_counter += 1
                
            yield sentence
        
if __name__ == '__main__':

    t0 = time.time()

    import argparse

    argparc = argparse.ArgumentParser(description='')

    mode = argparc.add_mutually_exclusive_group(required=True)
    mode.add_argument('-train', dest='train', action='store_true', help='run in training mode')
    mode.add_argument('-test', dest='test', action='store_true', help='run in test mode')
    mode.add_argument('-evaluate', dest='evaluate', action='store_true', help='run in evaluation mode')

    argparc.add_argument('-i', '--infile', dest='in_file', help='in file', required=True)
    argparc.add_argument('-sentence-limit', dest='sentence_limit', help='sentence limit', default='-1')
    argparc.add_argument('-e', '--epochs', dest='epochs', help='epochs', default='1')
    argparc.add_argument('-m', '--model', dest='model', help='model', default='model')
    argparc.add_argument('-o', '--output', dest='output_file', help='output file', default='output.txt')
    argparc.add_argument('-decrease-alpha', dest='decrease_alpha', action='store_true', help='decrease alpha', default=False)
    argparc.add_argument('-shuffle-sentences', dest='shuffle_sentences', action='store_true', help='shuffle sentences', default=False)
    argparc.add_argument('-batch-training', dest='batch_training', action='store_true', help='batch training', default=False)

    args = argparc.parse_args()

    d = depParser()
    if os.stat(args.in_file).st_size == 0:
        print "Input file is empty"
    else:
        if args.train:
            print "Running in training mode\n"
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