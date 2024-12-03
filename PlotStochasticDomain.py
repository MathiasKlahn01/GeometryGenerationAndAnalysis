# Import libraries
import numpy             as np
import matplotlib.pyplot as plt
from   matplotlib.colors import ListedColormap, BoundaryNorm
from   GeometryFunctions import *



# Define domain parameters 
Nx       = 640
Ny       = 640
gamma    = 0.5
lambdaxy = 10.0
rngSeed  = 1


# Generate the domain and plot it
Domain, nIter = generate_stochastic_domain(Nx, Ny, gamma, lambdaxy, rngSeed)


cmap = ListedColormap(['blue', 'yellow'])
norm = BoundaryNorm([0, 0.5, 1], cmap.N)

plt.figure(1)
plt.contourf(Domain, cmap=cmap, norm=norm)
#plt.colorbar()
plt.xlabel(r'$x/h$', fontsize=12)
plt.ylabel(r'$y/h$', fontsize=12)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0, Nx)
plt.ylim(0, Ny)
plt.savefig("Figures/StochasticDomainNx" + str(Nx) + "Ny" + str(Ny) + "gamma" + str(gamma) + "lambda" + str(lambdaxy) + "rngSeed" + str(rngSeed) + ".pdf")