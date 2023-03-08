import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
from numpy import linalg as LA

def create_wavelength():
    wavelength = np.zeros((1, 31))
    for idx in range(31):
        wavelength[0][idx] = 400 + 10 * idx
    #wavelength = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510,
    #              520, 520, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630,
    #              640, 650, 660, 670, 680, 690, 700]
    return wavelength

def plot_wavelength(data, wavelength):
    plt.clf
    plt.title("XYZ Color Matching Functions")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Value")
    plt.plot(wavelength[0,:], data['x'][0,:])
    plt.plot(wavelength[0,:], data['y'][0,:])
    plt.plot(wavelength[0,:], data['z'][0,:])
    plt.legend(['x', 'y', 'z'])
    plt.show()

def plot_lms(lms, wavelength):
    plt.clf
    plt.title("LMS Color Matching Functions")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Value")
    plt.plot(wavelength[0,:],lms[0,:])
    plt.plot(wavelength[0,:],lms[1,:])
    plt.plot(wavelength[0,:],lms[2,:])
    plt.legend(['l', 'm', 's'])
    plt.show()

def plot_illumination(data, wavelength):
    plt.clf
    plt.title("Illumination")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Value")
    plt.plot(wavelength[0,:],data['illum1'][0,:])
    plt.plot(wavelength[0,:],data['illum2'][0,:])
    plt.legend(['D65', 'Fluorescent'])
    plt.show()

# Load data.npy
data = np.load('.\CIE_data\data.npy', allow_pickle=True)[()]
# List keys of dataset
print(data.keys())
wavelength = create_wavelength()
#plot_wavelength(data, wavelength)

A_inv = np.array([[0.243,0.856,-0.044],
                 [-0.3910,1.1650,0.0870],
                 [0.01,-0.008,0.5630]])

cie_1931 = np.zeros((3,31))

cie_1931[0,:] = data['x'][0,:]
cie_1931[1,:] = data['y'][0,:]
cie_1931[2,:] = data['z'][0,:]

lms = np.matmul(A_inv,cie_1931)
#plot_lms(lms, wavelength)
plot_illumination(data, wavelength)

#plt.plot(X[0,:], X[1,:],'.')
#plt.axis('equal')
#plt.title(title)
#plt.show()