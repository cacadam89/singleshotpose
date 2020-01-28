import numpy as np 
import matplotlib.pyplot as plt
import pdb

data = np.load("/home/awc11/Documents/singleshotpose/backup/mslquad/costs.npz")
losses = data['training_losses']
N = len(losses)
plt.plot(range(N), losses, 'b-')
ax = plt.gca()
ax.set_title("Loss")
ax.axes.set_xlabel("itr... epoch??")
ax.axes.set_ylabel("loss")
plt.show()