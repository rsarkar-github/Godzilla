import numpy as np
import matplotlib.pyplot as plt


n = 61
nz = 61
x = np.load("G:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Data/linvel_gaussian_sol.npz")["arr_0"]
x = np.reshape(x, newshape=(nz, n, n))

scale = 1e-5
plt.imshow(np.real(x[:, :, int(n / 2)]), cmap="Greys", vmin=-scale, vmax=scale)
plt.colorbar()
plt.show()
