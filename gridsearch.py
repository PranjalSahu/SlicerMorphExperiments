import os
import numpy as np

'''
imageDiagonal = [800]#np.arange(800, 1001, 100)
StretchWeight = np.arange(0.002, 0.01, 0.002)
BendWeight    = np.arange(0.002, 0.01, 0.002)
GeometricWeight = np.arange(0.002, 0.01, 0.002)
MaxStep         = [0.001]#np.arange(0.001, 0.05, 0.005)
'''

imageDiagonal = [50]#np.arange(200, 601, 50)
StretchWeight = [0.0001]#np.arange(0.001, 0.01, 0.001)
BendWeight    = [0.0001]#np.arange(0.001, 0.01, 0.001)
GeometricWeight = [0.0001]#np.arange(0.001, 0.01, 0.001)
MaxStep         = [0.01]#np.arange(0.001, 0.05, 0.005)
totalSteps = [10]

for s in StretchWeight:
    for b in BendWeight:
        for g in GeometricWeight:
            for m in MaxStep:
                for i in imageDiagonal:
                    for t in totalSteps:
                        #os.system(f"python test_euc_displacement.py {i} {s} {b} {g} {m} {t}")
                        #os.system(f"python test_tsd_affine.py {i} {s} {b} {g} {m} {t}")
                        #os.system(f"python test_tsd_displacement.py {i} {s} {b} {g} {m} {t}")
                        os.system(f"python test_tsd_bspline.py {i} {s} {b} {g} {m} {t}")
                        #os.system(f"python test_jhct_displacement.py {i} {s} {b} {g} {m} {t}")
