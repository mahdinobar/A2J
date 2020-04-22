import numpy as np

idepth=np.loadtxt('/home/mahdi/HVR/hvr/data/iPad/set_1/iPad_1_Depth_1.txt') # meter
import matplotlib.pyplot as plt
import matplotlib

fig, ax = plt.subplots()
ax.imshow(idepth, cmap=matplotlib.cm.jet)
plt.show()

print('ended')