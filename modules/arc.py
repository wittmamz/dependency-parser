class Arc(object):

    # initialize token from a line in file:
    def __init__(self, s_id, a_id, head, dependent, relation):

        self.sparse_feat_vec = []
        self.s_id = s_id
        self.a_id = a_id
        self.head = head
        self.dependent = dependent
        self.relation = relation
        temp = head.t_id - dependent.t_id
        if temp < 0:
            self.length = (temp*-1)-1
            self.direction = "R"
        else:
            self.length = temp-1
            self.direction = "L"
        

        
    # create the sparse feature vector for this token (addin only applicable features):
    def createFeatureVector(self, feat_vec):
    
        if "hform_" + self.head.form + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hform_" + self.head.form + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "hpos_" + self.head.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hpos_" + self.head.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "hform_" + self.head.form + "_hpos_" + self.head.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hform_" + self.head.form + "_hpos_" + self.head.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
            
        if "dform_" + self.dependent.form + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["dform_" + self.dependent.form + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "dform_" + self.dependent.form + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["dform_" + self.dependent.form + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
            
        if "hform_" + self.head.form + "_dform_" + self.dependent.form + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hform_" + self.head.form + "_dform_" + self.dependent.form + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
            
        if "hform_" + self.head.form + "_hpos_" + self.head.pos + "_dform_" + self.dependent.form + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hform_" + self.head.form + "_hpos_" + self.head.pos + "_dform_" + self.dependent.form + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "hform_" + self.head.form + "_hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hform_" + self.head.form + "_hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction])    
            
        if "hform_" + self.head.form + "_dform_" + self.dependent.form + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hform_" + self.head.form + "_dform_" + self.dependent.form + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "hpos_" + self.head.pos + "_dform_" + self.dependent.form + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hpos_" + self.head.pos + "_dform_" + self.dependent.form + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        
        if "hform_" + self.head.form + "_hpos_" + self.head.pos + "_dform_" + self.dependent.form + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hform_" + self.head.form + "_hpos_" + self.head.pos + "_dform_" + self.dependent.form + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        

        between_pos = []
        if self.direction == "R":
            temp = self.head.next_token
            while temp.t_id != self.dependent.t_id:
                between_pos.append(temp.pos)
                temp = temp.next_token
        else:
            temp = self.head.prev_token
            while temp.t_id != self.dependent.t_id:
                between_pos.append(temp.pos)
                temp = temp.prev_token
        if len(between_pos) == 0:
            between_pos.append("$none$")
            
        if "hpos_" + self.head.pos + "_bpos_" + "_bpos_".join(between_pos) + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hpos_" + self.head.pos + "_bpos_" + "_bpos_".join(between_pos) + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        
        if self.head.next_token != None and self.dependent.next_token != None:
            if "hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_hpos+1_" + self.head.next_token.pos + "_dpos+1_" + self.dependent.next_token.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
                self.sparse_feat_vec.append(feat_vec["hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_hpos+1_" + self.head.next_token.pos + "_dpos+1_" + self.dependent.next_token.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        if self.head.next_token != None and self.dependent.prev_token != None:
            if "hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_hpos+1_" + self.head.next_token.pos + "_dpos-1_" + self.dependent.prev_token.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
                self.sparse_feat_vec.append(feat_vec["hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_hpos+1_" + self.head.next_token.pos + "_dpos-1_" + self.dependent.prev_token.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        if self.head.prev_token != None and self.dependent.next_token != None:
            if "hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_hpos-1_" + self.head.prev_token.pos + "_dpos+1_" + self.dependent.next_token.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
                self.sparse_feat_vec.append(feat_vec["hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_hpos-1_" + self.head.prev_token.pos + "_dpos+1_" + self.dependent.next_token.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        if self.head.prev_token != None and self.dependent.prev_token != None:
            if "hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_hpos-1_" + self.head.prev_token.pos + "_dpos-1_" + self.dependent.prev_token.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
                self.sparse_feat_vec.append(feat_vec["hpos_" + self.head.pos + "_dpos_" + self.dependent.pos + "_hpos-1_" + self.head.prev_token.pos + "_dpos-1_" + self.dependent.prev_token.pos + "_len_" + str(self.length) + "_dir_" + self.direction])

        # Uppercase

        #if self.uppercase:
            #self.sparse_feat_vec.append(feat_vec["uppercase"])

        #if self.capitalized:
            #self.sparse_feat_vec.append(feat_vec["capitalized"])

        # form

       
        # form length

        # the current form length:
        #if "current_word_len_" + str(len(current_token.form)) in feat_vec:
            #self.sparse_feat_vec.append(feat_vec["current_word_len_" + str(len(current_token.form))])

        # if applicable, the previous form length:
        #if previous_token:
            #if "prev_word_len_" + str(len(previous_token.form)) in feat_vec:
                #self.sparse_feat_vec.append(feat_vec["prev_word_len_" + str(len(previous_token.form))])

        # if applicable, the next token form length:
        #if next_token:
            #if "next_word_len_" + str(len(next_token.form)) in feat_vec:
                #self.sparse_feat_vec.append(feat_vec["next_word_len_" + str(len(next_token.form))])

        # position in sentence

        #if "position_in_sentence_" + str(t_id) in feat_vec:
            #self.sparse_feat_vec.append(feat_vec["position_in_sentence_" + str(t_id)])
            
        #for i in self.top_x:
            #if "prefix_" + current_token.form[:i] in feat_vec:
                #self.sparse_feat_vec.append(feat_vec["prefix_" + current_token.form[:i]])
            #if "suffix_" + current_token.form[-i:] in feat_vec:
                #self.sparse_feat_vec.append(feat_vec["suffix_" + current_token.form[-i:]])

            #if len(current_token.form) > i+1 and i > 2:

                # letter combinations in the word
                # if they don't overlap with pre- or suffixes
                #for j in range(i, len(current_token.form)-(i*2-1)):
                    #if "lettercombs_" + current_token.form[j:j+i] in feat_vec:
                        #self.sparse_feat_vec.append(feat_vec["lettercombs_" + current_token.form[j:j+i]])




