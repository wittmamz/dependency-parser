import modules.token as tk
import modules.arc as ar

def score(w, f):
    return sum([float(w[x]) for x in f])


def dfs(graph, start):
    visited, stack = [], [start]
    while stack:
        vertex = stack.pop()
        #if vertex not in visited:
        visited.append(vertex)
        temp = []
        for elem in graph[1]:
            if elem[1].head.t_id == vertex:
                temp.append(elem[1].dependent.t_id)
        for elem in temp:
            for ind in range(len(visited)):
                if visited[ind] == elem:
                    return [True, visited[ind:]]
        stack.extend(temp)
    return [False, visited]
    
def findCycle(graph):
    to_do = [x.t_id for x in graph[0] if x.t_id != 0]
    while len(to_do) > 0:
        temp = dfs(graph, to_do[0])
        if temp[0]:
            result = [[], []]
            for id in temp[1]:
                result[0].append([x for x in graph[0] if x.t_id == id][0])
            for elem in graph[1]:
                if len(set([elem[1].head.t_id, elem[1].dependent.t_id]).intersection(set([x.t_id for x in result[0]]))) == 2:
                    result[1].append(elem)
            return result
        else:
            to_do = [x for x in to_do if x not in temp[1]]
    return None
    
    

def contract(graph, cycle):
    result = [[x for x in graph[0] if x not in cycle[0]], [x for x in graph[1] if len(set([x[1].head.t_id, x[1].dependent.t_id]).intersection(set([y.t_id for y in cycle[0]]))) == 0]]
    cycle_token = tk.Token("CYCLE", graph[0][0].s_id, sorted([x.t_id for x in cycle[0]])[0])
    result[0].append(cycle_token)
    
    for token in [x for x in graph[0] if x not in cycle[0]]:
        try:
            temp = sorted([x for x in graph[1] if x[1].dependent.t_id == token.t_id and x[1].head.t_id in [y.t_id for y in cycle[0]]], key = lambda x: x[0])[-1]
            new_elem = [temp[0], ar.Arc(temp[1].s_id, temp[1].a_id, cycle_token, temp[1].dependent, temp[1].relation)]
            result[1].append(new_elem)
        except IndexError:
            pass
            
    for token in [x for x in graph[0] if x not in cycle[0]]:
        new_elem = None
        for elem in [x for x in graph[1] if x[1].head.t_id == token.t_id and x[1].dependent.t_id in [y.t_id for y in cycle[0]]]:
            new_score = elem[0] + sum([x[0] for x in cycle[1] if elem[1].dependent.t_id != x[1].dependent.t_id])
            if new_elem == None:
                new_elem = [new_score, ar.Arc(elem[1].s_id, elem[1].a_id, elem[1].head, cycle_token, elem[1].relation)]
            else:
                if new_score > new_elem[0]:
                    new_elem = [new_score, ar.Arc(elem[1].s_id, elem[1].a_id, elem[1].head, cycle_token, elem[1].relation)]

        result[1].append(new_elem)
    
    return result
            


def CLE(graph, weights=None, recursion_depth=0):
    if recursion_depth == 0:
        arcs_new = []
        for arc in graph[1]:
            arcs_new.append([score(weights, arc.sparse_feat_vec), arc])
        graph = [graph[0], arcs_new]
##    print str(recursion_depth) + ": graph"
##    for elem in graph[1]:
##        print str(elem[0]) + " - (" + str(elem[1].head.t_id) + ", " + str(elem[1].dependent.t_id) + ")" + " - " + str(elem[1].a_id)
##    print "-----------------------"    
    max_arcs = []
    for token in graph[0]:
        if token.t_id != 0:
            max_arc = []
            for elem in graph[1]:
                if elem[1].dependent.t_id == token.t_id:
                    #print "hi"
                    if len(max_arc) == 0:
                        max_arc = elem
                    else:
                        if elem[0] > max_arc[0]:
                            max_arc = elem
            max_arcs.append(max_arc)
    
    graph_a = [graph[0], max_arcs]
##    print str(recursion_depth) + ": graph_a"
##    for elem in graph_a[1]:
##        print str(elem[0]) + " - (" + str(elem[1].head.t_id) + ", " + str(elem[1].dependent.t_id) + ")" + " - " + str(elem[1].a_id)
##    print "-----------------------"
    cycle = findCycle(graph_a)
    if cycle == None:
        if recursion_depth == 0:
            return [graph_a[0], [x[1] for x in graph_a[1]]]
        else:
            return graph_a
    else:
##        print str(recursion_depth) + ": cycle"
##        for elem in cycle[1]:
##            print str(elem[0]) + " - (" + str(elem[1].head.t_id) + ", " + str(elem[1].dependent.t_id) + ")" + " - " + str(elem[1].a_id)
##        print "-----------------------"
        graph_c = contract(graph, cycle)
##        print str(recursion_depth) + ": graph_c"
##        for elem in graph_c[1]:
##            print str(elem[0]) + " - (" + str(elem[1].head.t_id) + ", " + str(elem[1].dependent.t_id) + ")" + " - " + str(elem[1].a_id)
##        print "-----------------------"
        y = CLE(graph_c, None, recursion_depth+1)
##        print str(recursion_depth) + ": graph"
##        for elem in graph[1]:
##            print str(elem[0]) + " - (" + str(elem[1].head.t_id) + ", " + str(elem[1].dependent.t_id) + ")" + " - " + str(elem[1].a_id)
##        print "-----------------------"
##        print str(recursion_depth) + ": cycle"
##        for elem in cycle[1]:
##            print str(elem[0]) + " - (" + str(elem[1].head.t_id) + ", " + str(elem[1].dependent.t_id) + ")" + " - " + str(elem[1].a_id)
##        print "-----------------------"
##        print str(recursion_depth) + ": y"
##        for elem in y[1]:
##            print str(elem[0]) + " - (" + str(elem[1].head.t_id) + ", " + str(elem[1].dependent.t_id) + ")" + " - " + str(elem[1].a_id)
##        print "-----------------------"
        for token in cycle[0]:
            #print token.t_id
            for elem in y[1]:
                #print str(elem[0]) + " - (" + str(elem[1].head.t_id) + ", " + str(elem[1].dependent.t_id) + ")" + " - " + str(elem[1].a_id)
                for elem2 in graph[1]:
                    #print str(elem2[0]) + " - (" + str(elem2[1].head.t_id) + ", " + str(elem2[1].dependent.t_id) + ")" + " - " + str(elem2[1].a_id)
                    if elem2[1].a_id == elem[1].a_id and elem2[1].dependent.t_id == token.t_id:
                        #print "yes"
                        if recursion_depth == 0:
                            arcs = []
                            for item in y[1]:
                                arcs.append([x[1] for x in graph[1] if x[1].a_id == item[1].a_id][0])
                            return [graph[0], arcs + [x[1] for x in cycle[1] if x[1].dependent.t_id != token.t_id]]
                        else:
                            arcs = []
                            for item in y[1]:
                                arcs.append([x for x in graph[1] if x[1].a_id == item[1].a_id][0])
                            return [graph[0], arcs + [x for x in cycle[1] if x[1].dependent.t_id != token.t_id]]
                        
                    
    
    
            
