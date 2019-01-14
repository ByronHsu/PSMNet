from utils import util
import numpy as np
disp = ['Synthetic/TLD' + str(i) + '.pfm' for i in range(10)]

for i in range(10):
    a = util.readPFM(disp[i])
    print(i, np.max(a))