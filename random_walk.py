import numpy as np
import random
import matplotlib.pyplot as plt

# Random coin flipping
FLIPPING = 1000
trajectory = []

for _ in range(FLIPPING):
    head = bool(random.getrandbits(1))
    trajectory.append(head)

x = np.arange(1, FLIPPING+1, 1)
agent = 0
y = []
for s in trajectory:
    if s == True:   agent+=1
    else:           agent-=1
    y.append(agent)

plt.plot(x,y)
plt.title("Random Walk Model")
plt.xlabel("random coin flipping")
plt.ylabel("agent's position (1D)")
plt.grid(which='major', color='k', linestyle = ":")
plt.show()