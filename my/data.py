import csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

pd_data = pd.read_csv("Bentonite.csv")
raw_data = pd_data.values

x = raw_data[:,0]
y = raw_data[:64,1]
Z = raw_data[:, 2:-1]
# print(x.shape)
# print(y)
# print(Z.shape)
my_cmap=plt.get_cmap('bwr')
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': '3d'})

X, Y = np.meshgrid(x, y)
# Z = raw_data[:64, 2:]
ax.plot_surface(X,Y,Z.T, cmap=my_cmap)
# for i in range(raw_data.shape[1]):
#     if not i in [0, 1]:
#         ax.scatter(x, y, raw_data[:64,i])   
    

plt.show()