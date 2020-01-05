import numpy as np 
import matplotlib.pylab as plt

x = np.array([[1],
              [2],
              [3]])

y = np.array([[4],
              [5],
              [6]])

print(x.T@y)

""" Exercise 2.2.1 """
p = np.array([100, 80, 50]).reshape((-1,1))
n = np.array([3, 4, 5]).reshape((-1,1))

w = p.T@n

print("w: {}".format(w))

print("=============================")

from sklearn.datasets import load_digits, load_iris
import matplotlib.gridspec as gridspec

digits = load_digits()
d1 = digits.images[0]
d2 = digits.images[10]
d3 = digits.images[1]
d4 = digits.images[11]
v1 = d1.reshape(64,1)
v2 = d2.reshape(64,1)
v3 = d3.reshape(64,1)
v4 = d4.reshape(64,1)

print("similarity b/w zeros: ", v1.T@v2)
print("similarity b/w ones: ", v3.T@v4)
print("similarity b/w zero and one: ", v1.T@v3)

""" Exercise 2.2.3 """
X = load_digits().data
first_image = digits.images[0].reshape(64,1)
print(first_image.T@X[0]) #1

#2: Similarities of all images
SoAI = np.dot(X, X.T)

X = load_iris().data
print(X[:,0].mean())


from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()

f, ax = plt.subplots(1,3)

ax[0].imshow(faces.images[6], cmap=plt.cm.bone)
ax[0].grid(False)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('image 1: $x_1$')


ax[1].imshow(faces.images[10], cmap=plt.cm.bone)
ax[1].grid(False)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('image 1: $x_2$')

new_face = 0.7 * faces.images[6] + 0.3*faces.images[10]
ax[2].imshow(new_face, cmap=plt.cm.bone)
ax[2].grid(False)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title('image 3: $x_new$')

# plt.show()


""" Exercise 2.4.6 """
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target



