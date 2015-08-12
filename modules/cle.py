import modules.token as token_class
import modules.arc as arc_class

# calculate the score for an arc based on the weight and feature vectors:
def score(w, f):

    return sum([float(w[x]) for x in f])

# depth-first search to find cycles in a graph:
def dfs(graph, start):

    visited, stack = [], [start]
    while stack:
        vertex = stack.pop()
        visited.append(vertex)
        temp = []
        for elem in graph[1]:
            if elem[1].head.t_id == vertex:
                temp.append(elem[1].dependent.t_id)
        for elem in temp:
            for ind in range(len(visited)):
                if visited[ind] == elem:
                    return [True, visited[ind:]] # cycle found and returned
        stack.extend(temp)
    return [False, visited] # no cycle, list of visited nodes (tokens) returned
 
# find a cycle in a graph: 
def findCycle(graph):

    to_do = [x.t_id for x in graph[0] if x.t_id != 0] # add all tokens to "to do" list (except ROOT node)
    while len(to_do) > 0:
        temp = dfs(graph, to_do[0]) # perform DFS cycle search
        if temp[0]: # cycle found
            result = [[], []] # sub-graph representing the cycle
            for id in temp[1]:
                result[0].append([x for x in graph[0] if x.t_id == id][0])
            for elem in graph[1]:
                if len(set([elem[1].head.t_id, elem[1].dependent.t_id]).intersection(set([x.t_id for x in result[0]]))) == 2:
                    result[1].append(elem)
            return result
        else: # no cycle found, update "to do" list with nodes (tokens) from other graph components (if they exist)
            to_do = [x for x in to_do if x not in temp[1]]
    return None # no cycles found

# contract a cycle in a graph:
def contract(graph, cycle):
    
    # add the graph without the cycle to result, then add a special cycle node to substitute the cycle:
    result = [[x for x in graph[0] if x not in cycle[0]], 
              [x for x in graph[1] if len(set([x[1].head.t_id, x[1].dependent.t_id]).intersection(set([y.t_id for y in cycle[0]]))) == 0]]
    cycle_token = token_class.Token("CYCLE", graph[0][0].s_id, sorted([x.t_id for x in cycle[0]])[0])
    result[0].append(cycle_token)
    
    # recompute arc scores:
    
    # arcs leaving the cycle:
    for token in [x for x in graph[0] if x not in cycle[0]]:
        try:
            temp = sorted([x for x in graph[1] if x[1].dependent.t_id == token.t_id and x[1].head.t_id in [y.t_id for y in cycle[0]]], 
                          key = lambda x: x[0])[-1]
            new_elem = [temp[0], arc_class.Arc(temp[1].s_id, temp[1].a_id, cycle_token, temp[1].dependent, temp[1].relation)]
            result[1].append(new_elem)
        except IndexError: # no arcs from cycle to current token
            pass
           
    # arcs entering the cycle:
    for token in [x for x in graph[0] if x not in cycle[0]]:
        new_elem = None
        for elem in [x for x in graph[1] if x[1].head.t_id == token.t_id and x[1].dependent.t_id in [y.t_id for y in cycle[0]]]:
            new_score = elem[0] + sum([x[0] for x in cycle[1] if elem[1].dependent.t_id != x[1].dependent.t_id])
            if new_elem == None:
                new_elem = [new_score, arc_class.Arc(elem[1].s_id, elem[1].a_id, elem[1].head, cycle_token, elem[1].relation)]
            else:
                if new_score > new_elem[0]:
                    new_elem = [new_score, arc_class.Arc(elem[1].s_id, elem[1].a_id, elem[1].head, cycle_token, elem[1].relation)]
        result[1].append(new_elem)
    
    return result

# Chu-Liu/Edmonds' algorithm:
def CLE(graph, weights=None, recursion_depth=0):
    
    # add scores to the arcs in the list of arcs:
    if recursion_depth == 0:
        arcs_new = []
        for arc in graph[1]:
            arcs_new.append([score(weights, arc.sparse_feat_vec), arc])
        graph = [graph[0], arcs_new] # update the graph with new list of arcs including scores
        
    # find the maximum arcs pointing to each token:
    max_arcs = []
    for token in graph[0]:
        if token.t_id != 0:
            max_arc = []
            for elem in graph[1]:
                if elem[1].dependent.t_id == token.t_id:
                    if len(max_arc) == 0:
                        max_arc = elem
                    else:
                        if elem[0] > max_arc[0]:
                            max_arc = elem
            max_arcs.append(max_arc)
    
    graph_a = [graph[0], max_arcs] # maximum spanning tree
    
    cycle = findCycle(graph_a) # check for cycle in maximum spanning tree
    
    if cycle == None: # done
        if recursion_depth == 0: # return only arcs, without the scores added in the beginning
            return [graph_a[0], [x[1] for x in graph_a[1]]]
        else:
            return graph_a
    else: # cycle found
        graph_c = contract(graph, cycle) # contract the cycle in the graph
        
        y = CLE(graph_c, None, recursion_depth+1) # call CLE again recursively, using the contracted graph
        
        # resolve the cycle:
        for token in cycle[0]:
            for elem in y[1]:
                for elem2 in graph[1]:
                    if elem2[1].a_id == elem[1].a_id and elem2[1].dependent.t_id == token.t_id: # found the maximum arc leading into the cycle
                    
                        # return the graph with the broken cycle (remove the arc in the cycle leading into "token"):
                        if recursion_depth == 0: # remove the scores from the list of arcs
                            arcs = []
                            for item in y[1]:
                                arcs.append([x[1] for x in graph[1] if x[1].a_id == item[1].a_id][0])
                            return [graph[0], arcs + [x[1] for x in cycle[1] if x[1].dependent.t_id != token.t_id]]
                        else:
                            arcs = []
                            for item in y[1]:
                                arcs.append([x for x in graph[1] if x[1].a_id == item[1].a_id][0])
                            return [graph[0], arcs + [x for x in cycle[1] if x[1].dependent.t_id != token.t_id]]