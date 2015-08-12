class Token(object):

    # initialize token from a line in file:
    def __init__(self, entries, s_id, t_id=None):
        self.next_token = None
        self.prev_token = None
        self.s_id = s_id
        #self.top_x = [2,3,4,5]
        
        if entries == "ROOT":
            self.t_id = 0
            self.form = "$ROOT$"
            self.lemma = "$ROOT$"
            self.pos = "$ROOT$"
        elif entries == "CYCLE":
            self.t_id = t_id
            self.form = "$CYCLE$"
            self.lemma = "$CYCLE$"
            self.pos = "$CYCLE$"
        else:
            self.t_id = int(entries[0])
            self.form = entries[1]
            self.lemma = entries[2]
            self.pos = entries[3]
            
        #self.uppercase = False
        #self.capitalized = False
        
    def setAdjacentTokens(self, prev_token, next_token):
        self.prev_token = prev_token
        self.next_token = next_token