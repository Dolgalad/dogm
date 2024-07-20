import sys
import networkx as nx
import matplotlib.pyplot as plt

graph_path = sys.argv[1]
sol_path = sys.argv[2]
pred_path = sys.argv[3]

# load graph
with open(graph_path, "r") as f:
    d = f.read().split("\n")
    #d = filter(None, d)
    d = [r.split(" ") for r in d]
    print(d)
    d = [[int(w) for w in r if len(w)>0] for r in d]
n = d[0][0]

g = nx.Graph()
for i in range(n):
    if len(d[i+1])==0:
        print("single node " , i)
        g.add_node(i)
    else:
        for j in range(len(d[i+1])):
            g.add_edge(i, d[i+1][j]-1)

# load optimal solution
with open(sol_path) as f:
    sol = f.read().split("\n")
    sol = filter(None, sol)
    sol = [int(w) for w in sol]
print(sol)

print(g.number_of_nodes(), n)

# load scores solution
with open(pred_path) as f:
    pred = f.read().split("\n")
    pred = list(filter(None, pred))
    print(pred)
    pred = [[float(w) for w in r.split(",")] for r in pred]

# nodecolors
nodecolors = []
nodecolors_pred = []
nodescores = []
for n in g.nodes:
    if sol[n]==0:
        nodecolors.append("#1F78b4")
    else:
        nodecolors.append("green")

    if pred[n][0]==0:
        nodecolors_pred.append("#1F78b4")
    else:
        nodecolors_pred.append("green")
    nodescores.append(pred[n][1])

pos = nx.drawing.layout.spring_layout(g)
fig,ax = plt.subplots(1,3)
nx.draw_networkx(g, pos, node_color=nodecolors, ax=ax[0])
nx.draw_networkx(g, pos, node_color=nodecolors_pred, ax=ax[1])
nx.draw_networkx(g, pos, node_color=nodescores, cmap=plt.cm.Reds, ax=ax[2])

plt.show()
