import matplotlib.pyplot as plt
import numpy as np
import time
import sys

plt.rcParams['text.usetex'] = True

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

filename = "build/train_loss.txt"
if len(sys.argv)>1:
    filename = sys.argv[1]
running_n = 10

fig,ax = plt.subplots(1,2)

#while True:
with open(filename) as f:
    d = f.read().split("\n")
    d = list(filter(None, d))
    d = [r.split(",") for r in d]

#print(d)
epochs = np.array([int(r[0]) for r in d])
train_loss = np.array([float(r[1]) for r in d])
val_loss = np.array([float(r[2]) for r in d])
train_times = np.array([float(r[3]) for r in d])
val_times = np.array([float(r[4]) for r in d])

ax[0].plot(epochs[:-running_n+1], running_mean(train_loss,running_n), label="train")
ax[0].plot(epochs[:-running_n+1], running_mean(val_loss,running_n), label="val")
ax[0].set_yscale("log")
ax[0].set_ylabel("BCELoss")
ax[0].set_xlabel("Epoch")
ax[0].grid(True)
ax[1].plot(epochs[:-running_n+1], running_mean(train_times,running_n)/800, label="train")
ax[1].plot(epochs[:-running_n+1], running_mean(val_times,running_n)/200, label="val")
ax[1].set_yscale("log")
ax[1].set_ylabel("Time ($\mu$s)")
ax[1].set_xlabel("Epoch")
ax[1].grid(True)

plt.legend()
plt.tight_layout()
#plt.xscale("log")
plt.show()
#plt.draw()
#plt.pause(0.0001)
##plt.show(block=False)
#
#time.sleep(15.)
