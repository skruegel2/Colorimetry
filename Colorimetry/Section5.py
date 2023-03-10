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

def gamma_correct(RGB, gamma):
    for row_idx in range(RGB.shape[0]):
        for col_idx in range(RGB.shape[1]):
            for color_idx in range(3):
                RGB[row_idx,col_idx,color_idx] = pow(RGB[row_idx,col_idx,color_idx],1/gamma)
    return RGB

def scale_image(RGB):
    RGB_int = np.zeros(RGB.shape,np.uint8)
    for row_idx in range(RGB.shape[0]):
        for col_idx in range(RGB.shape[1]):
            for color_idx in range(3):
                RGB_int[row_idx,col_idx,color_idx] = np.uint8(255*RGB[row_idx,col_idx,color_idx])

def plot_chromaticity(data,R_CIE_1931,G_CIE_1931,B_CIE_1931,R_709,G_709,B_709,
                      D65, EE):
    xchrom = data['x'][0,:] / (data['x'][0,:] + data['y'][0,:] +
                               data['z'][0,:] )
    ychrom = data['y'][0,:] / (data['x'][0,:] + data['y'][0,:] +
                               data['z'][0,:] )
    #plt.clf()
    plt.style.use('default')
#    plt.title("Pure Spectral Source Chromaticity")
    #plt.xlabel("Wavelength [nm]")
    plt.plot(xchrom, ychrom)
    # CIE
    x_cie = [R_CIE_1931[0],G_CIE_1931[0],B_CIE_1931[0],R_CIE_1931[0]]
    y_cie  = [R_CIE_1931[1],G_CIE_1931[1],B_CIE_1931[1],R_CIE_1931[1]]
    x_709 = [R_709[0], G_709[0], B_709[0], R_709[0]]
    y_709 = [R_709[1], G_709[1], B_709[1], R_709[1]]
    x_D65 = [D65[0]]
    y_D65 = [D65[1]]
    x_EE = [EE[0]]
    y_EE = [EE[1]]
    plt.plot(x_cie, y_cie,'y')
    plt.text(R_CIE_1931[0],R_CIE_1931[1],'R CIE')
    plt.text(G_CIE_1931[0],G_CIE_1931[1],'G CIE')
    plt.text(B_CIE_1931[0],B_CIE_1931[1],'B CIE')
    plt.plot(x_709, y_709,'g')
    plt.text(R_709[0],R_709[1],'R 709')
    plt.text(G_709[0],G_709[1],'G 709')
    plt.text(B_709[0],B_709[1],'B 709')
    plt.plot(x_D65, y_D65, 'co')
    plt.text(D65[0]-0.13,D65[1]-0.02,'D65 White')
    plt.plot(x_EE, y_EE, 'mo')
    plt.text(EE[0]+0.025, EE[1],'Equal Energy White')

x_dim = np.arange(0, 1, 0.005)
y_dim = np.arange(0, 1, 0.005)

#x, y = np.meshgrid(x_dim, y_dim, indexing='ij')
x, y = np.meshgrid(x_dim, y_dim)
z = 1 - x - y

M = [[0.640, 0.300, 0.150],
     [0.330, 0.600, 0.060],
     [0.030, 0.100, 0.790]]
RGB = np.zeros((200, 200, 3))
RGB = compute_RGB(RGB, x, y, z, M)
RGB = remove_negative(RGB)
RGB = gamma_correct(RGB,2.2)
R_CIE_1931 = [0.73467, 0.26533, 0.0]
G_CIE_1931 = [0.27376, 0.71741, 0.00883]
B_CIE_1931 = [0.16658, 0.00886, 0.82456]

R_709 = [0.640, 0.330, 0.030]
G_709 = [0.300, 0.600, 0.100]
B_709 = [0.150, 0.060, 0.790]
D65 = [0.3127, 0.3290, 0.3583]
EE = [0.3333, 0.3333, 0.3333]
data = np.load('.\CIE_data\data.npy', allow_pickle=True)[()]

plt.imshow(RGB,extent=[0,1,1,0])
plot_chromaticity(data,R_CIE_1931,G_CIE_1931,B_CIE_1931,R_709,G_709,B_709,
                  D65, EE)
plt.show()
temp = 5