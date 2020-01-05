import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import load_iris

iris = load_iris()
x2 = np.array([4.9, 3.0, 1.4, 0.2])
print("Exercise 2.1.1:")
print("x2: {}".format(x2))
print("x2_shape: {}".format(x2.shape))
print("reshape x2 as (1,-1): \n{}".format(x2.reshape((1,-1))))
print("reshape x2 as (-1,1): \n{}".format(x2.reshape((-1,1))))

print("---------- ========================== ---------")

from sklearn.datasets import load_digits

digits = load_digits()
samples = [0, 10, 20, 30, 1, 11, 21, 31] # Indices of samples
d = [] # Store the samples of digits
for i in range(8):
    d.append(digits.images[samples[i]])

plt.figure(figsize=(8,2))
for i in range(8):
    plt.subplot(1, 8, i+1) # subplot(rows, columns, index, **kwarg)
    plt.imshow(d[i], interpolation='nearest', cmap=plt.cm.bone_r)
    plt.grid(False); plt.xticks([]); plt.yticks([])
    plt.title('image {}'.format(i+1))
plt.suptitle('Images of zero and one')
plt.tight_layout()
# plt.show()

v = []
for i in range(8):
    v.append(d[i].reshape(64,1)) # Unfolding each data and contain to 'v'

plt.figure(figsize=(8,3))
for i in range(8):
    plt.subplot(1, 8, i+1) # subplot(rows, columns, index, **kwarg)
    plt.imshow(v[i], aspect=0.4,
               interpolation='nearest', cmap=plt.cm.bone_r)
    plt.grid(False); plt.xticks([]); plt.yticks([])
    plt.title('Vector {}'.format(i+1))
plt.suptitle('Images of vectorized zero and one', y=1.05)
plt.tight_layout()
# plt.show()

print("Exercise 2.1.2")
X = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.9, 3.0, 1.4, 0.2]])
print("X: \n{}".format(X))              

print("---------- ============= TENSOR ============= ---------")
from scipy import misc

img_rgb = misc.face()
print(img_rgb.shape)

plt.subplot(221)
plt.imshow(img_rgb, cmap = plt.cm.gray)
plt.axis('off')
plt.title('RGB image')

plt.subplot(222)
plt.imshow(img_rgb[:,:,0], cmap=plt.cm.gray)
plt.axis('off')
plt.title('Red channel')

plt.subplot(223)
plt.imshow(img_rgb[:,:,1], cmap=plt.cm.gray)
plt.axis('off')
plt.title('Green channel')

plt.subplot(224)
plt.imshow(img_rgb[:,:,2], cmap=plt.cm.gray)
plt.axis('off')
plt.title('Blue channel')

# plt.show()

print("Exercise 2.1.3")
X = np.asarray(load_iris())
print("X_transpose: \n{}".format(X.T))
print("(X.T).T == X: {}".format(X.T.T == X))


print("Exercise 2.1.4")

zero_vector = np.zeros((5,1))
one_vector = np.ones((5,1))
square_matrix = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
identity_matrix = np.identity(5)
symmetric_matrix = np.array([[1, 2, 3],
                             [2, 4, 5],
                             [3, 5, 6]])                        

print(zero_vector)
print(one_vector)
print(square_matrix)
print(identity_matrix)
print(symmetric_matrix)