class Arc(object):

    # initialize arc:
    def __init__(self, s_id, a_id, head, dependent, relation):

        self.sparse_feat_vec = []
        self.s_id = s_id # sentence ID
        self.a_id = a_id # arc ID
        self.head = head
        self.dependent = dependent
        self.relation = relation
        temp = head.t_id - dependent.t_id
        if temp < 0:
            self.length = (temp * -1) - 1
            self.direction = "R"
        else:
            self.length = temp - 1
            self.direction = "L"
        

        
    # create the sparse feature vector for this arc (adding only applicable features):
    def createFeatureVector(self, feat_vec):
        
        # unary head features:
        if "hform_" + self.head.form + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hform_" + self.head.form + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "hpos_" + self.head.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hpos_" + self.head.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "hform_" + self.head.form + "_hpos_" + self.head.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["hform_" + self.head.form + "_hpos_" + self.head.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
            
        # unary dependent features:
        if "dform_" + self.dependent.form + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["dform_" + self.dependent.form + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
        if "dform_" + self.dependent.form + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction in feat_vec:
            self.sparse_feat_vec.append(feat_vec["dform_" + self.dependent.form + "_dpos_" + self.dependent.pos + "_len_" + str(self.length) + "_dir_" + self.direction])
            
        # binary features (head and dependent):
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
        
        # feature taking into account the POS of all tokens between head and dependent:
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
        
        # features taking into account the token before/after the head/dependent:
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