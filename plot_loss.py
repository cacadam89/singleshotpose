import numpy as np 
import matplotlib.pyplot as plt
import pdb

data = np.load("./backup/mslquad/costs.npz")
losses = data['training_losses']
N = len(losses)
plt.plot(range(N), losses, 'b-')
ax = plt.gca()
ax.set_title("Loss")
ax.axes.set_xlabel("itr")
ax.axes.set_ylabel("loss")
plt.show(block=False)

test_acc = data['testing_accuracies']
pix_err = data['testing_errors_pixel']
ang_err = data['testing_errors_angle']
plt.figure()
plt.plot(range(len(test_acc)), test_acc, 'b-')
plt.plot(range(len(pix_err)), pix_err, 'r-')
plt.plot(range(len(ang_err)), ang_err, 'm-')
plt.legend(['acc', 'pix err', 'ang err'])
ax = plt.gca()
ax.set_title("Errors / Accuracies")
ax.axes.set_xlabel("epoch")
plt.show()
