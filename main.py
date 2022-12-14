# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Metal optical constants
cu_n = pd.read_csv("copper_n.csv")
cu_k = pd.read_csv("copper_k.csv")

# Drop values outside of visible range
cu_n.drop(cu_n[ (cu_n.wl <= 0.38) | (cu_n.wl >= 0.740) ].index , inplace=True)
cu_k.drop(cu_k[ (cu_k.wl <= 0.38) | (cu_k.wl >= 0.740) ].index , inplace=True)

ag_n = pd.read_csv("silver_n.csv")
ag_k = pd.read_csv("silver_k.csv")

# Drop values outside of visible range
ag_n.drop(ag_n[ (ag_n.wl <= 0.38) | (ag_n.wl >= 0.740) ].index , inplace=True)
ag_k.drop(ag_k[ (ag_k.wl <= 0.38) | (ag_k.wl >= 0.740) ].index , inplace=True)

au_n = pd.read_csv("gold_n.csv")
au_k = pd.read_csv("gold_k.csv")

# Drop values outside of visible range
au_n.drop(au_n[ (au_n.wl <= 0.38) | (au_n.wl >= 0.740) ].index , inplace=True)
au_k.drop(au_k[ (au_k.wl <= 0.38) | (au_k.wl >= 0.740) ].index , inplace=True)

# Standard illumination at midday in northwest europe
std_illum = pd.read_csv("std_illum.csv")

# Color matching function
cmf = pd.read_csv("color_matching_functions.txt",sep='\t')

# Function to compute X,Y,Z
def compute(wl,n,k):
    # List of complex refractive indicies for all wavelengths
    index = np.vectorize(complex)(n, k)

    # List of power reflection coefficients for all wavelengths
    R = np.abs((index-1)/(index+1))**2

    # interpolate std_illum onto metal wavelengths
    illum = interpolate.interp1d(std_illum.iloc[:,0]/1000,std_illum.iloc[:,1])
    stdi = illum(wl)

    # Interpolate the color mapping onto metal wavelengths
    cm_x = interpolate.interp1d(cmf.iloc[:,0]/1000,cmf.X)
    x_i=cm_x(wl)
    
    cm_y = interpolate.interp1d(cmf.iloc[:,0]/1000,cmf.Y)
    y_i=cm_y(wl)

    cm_z = interpolate.interp1d(cmf.iloc[:,0]/1000,cmf.Z)
    z_i=cm_z(wl)

    # Plot reflectances
    plt.plot(wl,R)

    N = np.sum(stdi * y_i)

    # tristumulus values
    X = np.sum(R*stdi*x_i)/N
    Y = np.sum(R*stdi*y_i)/N
    Z = np.sum(R*stdi*z_i)/N

    # xy values
    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    z = Z/(X+Y+Z)

    # Conversion to RGB
    # Making a vector [X,Y,Z]

    XYZ = np.array([[X,Y,Z]]).T

    # Making the transformation matrix

    transform = np.matrix([[3.2406,-1.5372,-0.4986],[-0.9689,1.8758,0.0415],[0.0557,-0.2040,1.0570]])

    # Linear
    lin = np.matmul(transform,XYZ).ravel()

    # Gamma correction

    R = 0 
    G = 0
    B = 0

    print(lin[0,0])

    if lin[0,0] <= 0.0031308:
        R = 255*1292*lin[0,0]
    if lin[0,0] > 0.0031308:
        R = 255*1.055*np.power(lin[0,0],1/2.4) - 0.055

    if lin[0,1] <= 0.0031308:
        G = 255*1292*lin[0,1]
    if lin[0,1] > 0.0031308:
        G = 255*1.055*np.power(lin[0,1],1/2.4) - 0.055
    
    if lin[0,2] <= 0.0031308:
        B = 255*1292*lin[0,2]
    if lin[0,2] > 0.0031308:
        B = 255*1.055*np.power(lin[0,2],1/2.4) - 0.055

    return x,y, (R,G,B)

gold = compute(au_n.wl.tolist(),au_n.n.tolist(),au_k.k.tolist())
silver = compute(ag_n.wl.tolist(),ag_n.n.tolist(),ag_k.k.tolist())
copper = compute(cu_n.wl.tolist(),cu_n.n.tolist(),cu_k.k.tolist())

print(gold[2])
print(silver[2])
print(copper[2])

#plt.show()

# Results:
'''
Temperature:
Gold: 5465K
Silver: 11564K
Copper: 7280K

RGB:
Gold: (244.1493043715822, 238.876739792637, 213.60391676565922)
Silver: (233.8574692807422, 270.55393993966544, 316.88342571904474)
Copper: (232.45619279764685, 223.31875413765565, 244.99512360587357)
'''

