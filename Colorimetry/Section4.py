import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
from numpy import linalg as LA

def create_wavelength():
    wavelength = np.zeros((1, 31))
    for idx in range(31):
        wavelength[0][idx] = 400 + 10 * idx
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

def compute_XYZ_D65(I, data, wavelength):
    x = data['x'][0,:]
    y = data['y'][0,:]
    z = data['z'][0,:]
    XYZ = np.zeros((I.shape[0],I.shape[1], 3))
    XYZ[:,:,0] = np.dot(I,x)
    XYZ[:,:,1] = np.dot(I,y)
    XYZ[:,:,2] = np.dot(I,z)
    return XYZ

def compute_K(RGB_709, XYZ_WP):
    K = np.matmul(np.linalg.inv(RGB_709),XYZ_WP)
    return K

def compute_M(RGB_709, K):
    diag_K = np.diag(K)
    print(diag_K)
    M = np.matmul(RGB_709,diag_K)
    return M
# Section 4
# Load data.npy
data = np.load('./CIE_data/data.npy', allow_pickle=True)[()]

# Load reflect.npy
reflect_data = np.load('./reflect/reflect.npy', allow_pickle=True)[()]
wavelength = create_wavelength()
# Compute I
I = compute_I_D65(data, reflect_data, wavelength)
XYZ = compute_XYZ_D65(I, data, wavelength)

RGB_709 = [[0.640, 0.300, 0.150],
           [0.330, 0.600, 0.060],
           [0.030, 0.100, 0.790]]
D65_WP = [0.3127, 0.3290, 0.3583]
XYZ_WP = [0.3127/0.3290, 1, 0.3583/0.3290]
K = compute_K(RGB_709, XYZ_WP)
M = compute_M(RGB_709, K)
print(M)
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


#D65 = [0.3127, 0.3290, 0.3583]
#EE = [0.3333, 0.3333, 0.3333]
#plot_chromaticity(data,R_CIE_1931,G_CIE_1931,B_CIE_1931,R_709,G_709,B_709,
#                  D65, EE)

