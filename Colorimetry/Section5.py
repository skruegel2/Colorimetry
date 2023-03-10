import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image


# Section 5
def compute_RGB(RGB, x, y, z, M):
    for col_idx in range(RGB.shape[0]):
        for row_idx in range(RGB.shape[1]):
                RGB[col_idx,row_idx,:] = np.matmul(np.linalg.inv(M),[x[col_idx, row_idx], y[col_idx, row_idx], z[col_idx, row_idx]])
    return RGB
def remove_negative(RGB):
    for col_idx in range(RGB.shape[0]):
        for row_idx in range(RGB.shape[1]):
                if (RGB[col_idx, row_idx, 0] < 0 or
                    RGB[col_idx, row_idx, 1] < 0 or
                    RGB[col_idx, row_idx, 2] < 0):
                    RGB[col_idx, row_idx, 0] = 1
                    RGB[col_idx, row_idx, 1] = 1
                    RGB[col_idx, row_idx, 2] = 1
    return RGB

x_dim = np.arange(0, 1, 0.005)
y_dim = np.arange(0, 1, 0.005)

x, y = np.meshgrid(x_dim, y_dim, indexing='ij')
z = 1 - x - y

M = [[0.640, 0.300, 0.150],
     [0.330, 0.600, 0.060],
     [0.030, 0.100, 0.790]]
RGB = np.zeros((200, 200, 3))
RGB = compute_RGB(RGB, x, y, z, M)
RGB = remove_negative(RGB)
plt.imshow(RGB)
plt.show()
temp = 5