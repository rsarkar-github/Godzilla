import numpy as np
import matplotlib.pyplot as plt


n = 101
nz = 101

# Load solution
x = np.load("G:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Data/linvel_gaussian_sol_test1_alpha0.01.npz")["arr_0"]
# x = np.load("G:/Research/Freq-Domain/Godzilla/Python/IntegralEquation/Data/constvel_gaussian_sol_test1_c2.5.npz")["arr_0"]

scale = 1e-4
x = np.reshape(x, newshape=(nz, n))
fig = plt.figure()
plt.imshow(np.real(x), cmap="Greys", vmin=-scale, vmax=scale)
plt.grid(True)
plt.title("Real")
plt.colorbar()
plt.show()
