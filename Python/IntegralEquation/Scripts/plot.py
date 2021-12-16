import numpy as np
import matplotlib.pyplot as plt


n = 51
nz = 51
x = np.load("G:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Data/linvel_gaussian_solx.npz")["arr_0"]

scale = 1e-3
plt.imshow(np.real(x[:, :, int(n / 2)]), cmap="Greys", vmin=-scale, vmax=scale)
plt.colorbar()
plt.show()
