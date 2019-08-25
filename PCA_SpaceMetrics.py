# Partially copied from: Gaël Varoquaux
# Modified for ABCD dataset by Adam Pines

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmath as math
import sys
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

filenames = os.listdir('/Users/pinesa/Desktop/ABCDworkshop/PT_PCAS/')
pathname = '/Users/pinesa/Desktop/ABCDworkshop/PT_PCAS/'

# Size of list = 3777
valuesbw = myList = [None] * 3877
valuesor = myList = [None] * 3877
for i in range(3877):
    df=np.loadtxt(pathname + filenames[i],delimiter=',')
    # left caudalmiddlefrontal
    a,b,c,d=df[3]
    # right caudalmiddlefrontal
    x,y,z,q=df[37]
    
    # Distance between caudmidfronts, measure of asymmetry in PCA space
    distancebw=[np.sqrt((a-x)**2+(b-y)**2+(c-z)**2)]
    valuesbw[i]=distancebw
    print(distancebw)
    
    # Summed distance from origin caudmidfronts, measure of broad alignment with PCs
    distancefromor=[(np.sqrt((a)**2+(b)**2+(c)**2))+(np.sqrt((x)**2+(y)**2+(z)**2))]
    valuesor[i]=distancefromor
    print(distancefromor)

# Combine values and filenames for csv
allvals=np.c_[filenames[0:3777],valuesbw[0:3777],valuesor[0:3777]]

# Save
np.savetxt('fancyvalues', allvals, delimiter=',',fmt='%s')


