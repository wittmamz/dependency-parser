import modules.token as tk
import modules.arc as ar

from modules.cle import CLE

graph = [[tk.Token("ROOT", 0),
      tk.Token(["1", "1", "1", "1"], 0),
      tk.Token(["2", "2", "2", "2"], 0),
      tk.Token(["3", "3", "3", "3"], 0),
      tk.Token(["4", "1", "1", "1"], 0),
      tk.Token(["5", "1", "1", "1"], 0),
      tk.Token(["6", "1", "1", "1"], 0),
      tk.Token(["7", "1", "1", "1"], 0)], []]

graph[1].append([30.0, ar.Arc(0, 1, graph[0][2], graph[0][1], None)])
graph[1].append([30.0, ar.Arc(0, 2, graph[0][3], graph[0][2], None)])
graph[1].append([30.0, ar.Arc(0, 3, graph[0][3], graph[0][5], None)])
graph[1].append([30.0, ar.Arc(0, 4, graph[0][5], graph[0][4], None)])
graph[1].append([30.0, ar.Arc(0, 5, graph[0][4], graph[0][3], None)])
graph[1].append([30.0, ar.Arc(0, 6, graph[0][6], graph[0][7], None)])
graph[1].append([30.0, ar.Arc(0, 7, graph[0][7], graph[0][6], None)])
graph[1].append([20.0, ar.Arc(0, 8, graph[0][1], graph[0][3], None)])
graph[1].append([20.0, ar.Arc(0, 9, graph[0][5], graph[0][7], None)])

taken = [(2,1), (3,2), (3,5), (5,4), (4,3), (6,7), (7,6), (1,3), (5,7)]

counter2 = 10
for ind in range(len(graph[0])-1):
        for ind2 in range(ind+1,len(graph[0])):
        if (graph[0][ind].t_id, graph[0][ind2].t_id) not in taken:
                graph[1].append([10.0, ar.Arc(0, counter2, graph[0][ind], graph[0][ind2], None)])
                counter2 += 1
            if ind > 0 and (graph[0][ind2].t_id, graph[0][ind].t_id) not in taken:
                    graph[1].append([10.0, ar.Arc(0, counter2, graph[0][ind2], graph[0][ind], None)])
                    counter2 += 1

test = CLE(graph, None, 1)[1]

for elem in test:
    print str(elem[0]) + " - (" + str(elem[1].head.t_id) + ", " + str(elem[1].dependent.t_id) + ")"
    
