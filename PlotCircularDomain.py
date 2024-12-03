# Import libraries
import numpy             as np
import matplotlib.pyplot as plt
from   GeometryFunctions import *
from   matplotlib.colors import ListedColormap, BoundaryNorm


# Define domain parameters 
Nx      = 640
Ny      = 640
r       = 20.0   # r = 20.0 corrsponds to h = r/20, i.e. h_nondim = 0.05
minDist = 2.0    # distMin = 2.0 corresponds to distMin = r/10, i.e. distMin_nondim = 0.1 (at least when r = 20.0...)
por     = 0.5
NTry    = 1000
rngSeed = 1


# Generate the domain and plot it
Domain = generate_circular_domain(Nx, Ny, r, minDist, por, NTry, rngSeed)

x = np.linspace(0, Nx, Nx)
y = np.linspace(0, Ny, Ny)
X, Y = np.meshgrid(x, y)

cmap = ListedColormap(['blue', 'yellow'])
norm = BoundaryNorm([0, 0.5, 1], cmap.N)

plt.figure(1)
plt.contourf(X, Y, Domain, cmap=cmap, norm=norm)
plt.xlabel(r'$x/h$', fontsize=12)
plt.ylabel(r'$y/h$', fontsize=12)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0, Nx)
plt.ylim(0, Ny)
plt.savefig("Figures/CircularDomainNx" + str(Nx) + "Ny" + str(Ny) + "r" + str(r) + "minDist" + str(minDist) + "rngSeed" + str(rngSeed) + ".pdf")


