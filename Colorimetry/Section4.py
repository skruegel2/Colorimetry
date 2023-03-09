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

def plot_chromaticity(data,R_CIE_1931,G_CIE_1931,B_CIE_1931,R_709,G_709,B_709,
                      D65, EE):
    xchrom = data['x'][0,:] / (data['x'][0,:] + data['y'][0,:] +
                               data['z'][0,:] )
    ychrom = data['y'][0,:] / (data['x'][0,:] + data['y'][0,:] +
                               data['z'][0,:] )
    plt.clf()
    plt.style.use('default')
    plt.title("Pure Spectral Source Chromaticity")
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
    plt.show()

def compute_I_D65(data, reflect_data,wavelength):
    illum = data['illum1']
    reflect = reflect_data['R']
    I = reflect*illum
    return I

def compute_I_xyz_D65(I, data, wavelength):
    x = data['x'][0,:]
    y = data['y'][0,:]
    z = data['z'][0,:]
    I_xyz = np.zeros((I.shape[0],I.shape[1], 3))
    I_xyz[:,:,0] = np.dot(I,x)
    I_xyz[:,:,1] = np.dot(I,y)
    I_xyz[:,:,2] = np.dot(I,z)
    return I_xyz


# Section 4
# Load data.npy
data = np.load('./CIE_data/data.npy', allow_pickle=True)[()]

# Load reflect.npy
reflect_data = np.load('./reflect/reflect.npy', allow_pickle=True)[()]
wavelength = create_wavelength()
# Compute I
I = compute_I_D65(data, reflect_data, wavelength)
I_xyz = compute_I_xyz_D65(I, data, wavelength)


#wavelength = create_wavelength()
##plot_wavelength(data, wavelength)

#A_inv = np.array([[0.243,0.856,-0.044],
#                 [-0.3910,1.1650,0.0870],
#                 [0.01,-0.008,0.5630]])

#cie_1931 = np.zeros((3,31))

#cie_1931[0,:] = data['x'][0,:]
#cie_1931[1,:] = data['y'][0,:]
#cie_1931[2,:] = data['z'][0,:]



#lms = np.matmul(A_inv,cie_1931)
##plot_lms(lms, wavelength)
##plot_illumination(data, wavelength)
## Section 3
#R_CIE_1931 = [0.73467, 0.26533, 0.0]
#G_CIE_1931 = [0.27376, 0.71741, 0.00883]
#B_CIE_1931 = [0.16658, 0.00886, 0.82456]

#R_709 = [0.640, 0.330, 0.030]
#G_709 = [0.300, 0.600, 0.100]
#B_709 = [0.150, 0.060, 0.790]
#D65 = [0.3127, 0.3290, 0.3583]
#EE = [0.3333, 0.3333, 0.3333]
#plot_chromaticity(data,R_CIE_1931,G_CIE_1931,B_CIE_1931,R_709,G_709,B_709,
#                  D65, EE)

