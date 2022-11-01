import numpy as np
from matplotlib import pyplot as plt
x = np.linspace(1, 100, 1000)
y = np.log(x)
y1 = np.sin(x)
fig = plt.figure("Line plot")
legendFig = plt.figure("Legend plot")
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, c="red")
line2, = ax.plot(x, y1, c="blue")
legendFig.legend([line1, line2], ["Angular momentum", "Polarization"], loc='center')
legendFig.savefig('legend.png')