import os
import sys
from multiprocessing import Pool

n_p = 12
solver_path="../../KaMIS/deploy/weighted_branch_reduce"

dataset_path=sys.argv[1]
if len(sys.argv)>2:
    n_p = int(sys.argv[2])

cmd_list = []

for f in os.listdir(dataset_path):
    if f.endswith(".graph"):
        graph_path = os.path.join(dataset_path, f)
        sol_file = graph_path.replace(".graph", ".sol")
        cmd_list.append(f"{solver_path} {graph_path} --output={sol_file}")
    
with Pool(n_p) as p:
    p.map(os.system, cmd_list)
