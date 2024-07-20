import matplotlib.pyplot as plt
import numpy as np
import time
import sys

plt.rcParams['text.usetex'] = True

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

filename = "test_output.csv"
if len(sys.argv)>1:
    filename = sys.argv[1]

fig,ax = plt.subplots(2,2)

with open(filename) as f:
    d = f.read().split("\n")
    d = list(filter(None, d))
    d = [r.split(",") for r in d]

#print(d)
graphs = np.array([r[0] for r in d])
opt_val = np.array([float(r[1]) for r in d])
opt_time = np.array([float(r[2]) for r in d])
greedy_val = np.array([float(r[3]) for r in d])
greedy_time = np.array([float(r[4]) for r in d])
simple_greedy_val = np.array([float(r[5]) for r in d])

val_bins_min = min(opt_val.min(), greedy_val.min(), simple_greedy_val.min())
val_bins_max = max(opt_val.max(), greedy_val.max(), simple_greedy_val.max())
val_bins = np.linspace(val_bins_min, val_bins_max, 40)

time_bins_min = min(opt_time.min(), greedy_time.min())
time_bins_max = max(opt_time.max(), greedy_time.max())
time_bins = np.linspace(time_bins_min, time_bins_max, 80)

ax[0,0].hist(opt_val, val_bins, alpha=0.5, label="opt")
ax[0,0].hist(greedy_val, val_bins, alpha=0.5, label="greedy")
ax[0,0].hist(simple_greedy_val, val_bins, alpha=0.5, label="rgreedy")
ax[0,0].set_xlabel("Objective value")
ax[0,0].legend()

ax[0,1].hist(opt_time, time_bins, alpha=0.5, label="opt")
ax[0,1].hist(greedy_time, time_bins, alpha=0.5, label="greedy")
ax[0,1].set_xlabel("Solve time ($\mu$s)")

ax[1,0].hist( 100 * (opt_val - greedy_val) / opt_val , bins=50)
ax[1,0].axvline( np.mean(100*(opt_val-greedy_val)/opt_val), 0, 1, c="r")
ax[1,0].set_xlabel("Gap (\%)")

ax[1,1].hist( opt_time / greedy_time , bins=50)
ax[1,1].axvline( np.mean(opt_time/greedy_time), 0, 1, c="r")
ax[1,1].set_xlabel("Execution time ratio (\%)")

plt.tight_layout()

plt.show()
