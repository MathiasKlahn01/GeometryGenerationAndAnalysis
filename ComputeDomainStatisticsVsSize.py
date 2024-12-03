# Import libraries
import numpy             as np
import matplotlib.pyplot as plt
from   GeometryFunctions import *


# Define parameters used for both types of domain
NStat = 100
size  = 400

# Define parameters related to the circular obstacle domains
r       = 20.0   # r = 20.0 corrsponds to h = r/20, i.e. h_nondim = 0.05
minDist = 2.0    # distMin = 2.0 corresponds to distMin = r/10, i.e. distMin_nondim = 0.1 (at least when r = 20.0...)
por     = 0.5
NTry    = 1000

# Define parameters related to the stochastic domains
gamma    = 0.5
lambdaxy = 20.0


# Compute the statistics of the two types of domain                         
porosityCirc      = np.zeros(NStat);   porosityStoch      = np.zeros(NStat)
surfaceLengthCirc = np.zeros(NStat);   surfaceLengthStoch = np.zeros(NStat)
meanPoreWidthCirc = np.zeros(NStat);   meanPoreWidthStoch = np.zeros(NStat)
for rngSeed in range(NStat):
    print("rngSeed = ", rngSeed)
    # Circular obstacle domain
    DomainCirc = generate_circular_domain(size, size, r, minDist, por, NTry, rngSeed)
    porosityCirc[rngSeed]      = compute_porosity(DomainCirc)
    surfaceLengthCirc[rngSeed] = compute_surface_length(DomainCirc)
    meanPoreWidthCirc[rngSeed] = compute_mean_pore_width(DomainCirc)

    # Stochastic domain
    DomainStoch, nIter = generate_stochastic_domain(size, size, gamma, lambdaxy, rngSeed)
    porosityStoch[rngSeed]      = compute_porosity(DomainStoch)
    surfaceLengthStoch[rngSeed] = compute_surface_length(DomainStoch)
    meanPoreWidthStoch[rngSeed] = compute_mean_pore_width(DomainStoch)


# Write the result to file
np.save('Figures/Data/porosityCircNstat' + str(NStat) + 'size' + str(size) + \
                             'r' + str(r) + 'minDist' + str(minDist) + 'por' + str(por) + 'NTry' + str(NTry) + '.npy', porosityCirc)
np.save('Figures/Data/surfaceLengthCircNstat' + str(NStat) + 'size' + str(size) + \
                             'r' + str(r) + 'minDist' + str(minDist) + 'por' + str(por) + 'NTry' + str(NTry) + '.npy', surfaceLengthCirc)
np.save('Figures/Data/meanPoreWidthCircNstat' + str(NStat) + 'size' + str(size) + \
                             'r' + str(r) + 'minDist' + str(minDist) + 'por' + str(por) + 'NTry' + str(NTry) + '.npy', meanPoreWidthCirc)


np.save('Figures/Data/porosityStochNstat' + str(NStat) + 'size' + str(size) + \
                             'gamma' + str(gamma) + 'lambdaxy' + str(lambdaxy) + '.npy', porosityStoch)
np.save('Figures/Data/surfaceLengthStochNstat' + str(NStat) + 'size' + str(size) + \
                             'gamma' + str(gamma) + 'lambdaxy' + str(lambdaxy) + '.npy', surfaceLengthStoch)
np.save('Figures/Data/meanPoreWidthStochNstat' + str(NStat) + 'size' + str(size) + \
                             'gamma' + str(gamma) + 'lambdaxy' + str(lambdaxy) + '.npy', meanPoreWidthStoch)



""""
print("mean(porosityCirc) = ", np.mean(porosityCirc))
print("mean(porosityStoch) = ", np.mean(porosityStoch))
print("")
print("std(porosityCirc) = ", np.std(porosityCirc))
print("std(porosityStoch) = ", np.std(porosityStoch))
print("")
print("mean(surfaceLengthCirc) = ", np.mean(surfaceLengthCirc))
print("mean(surfaceLengthStoch) = ", np.mean(surfaceLengthStoch))
print("")
print("std(surfaceLengthCirc) = ", np.std(surfaceLengthCirc))
print("std(surfaceLengthStoch) = ", np.std(surfaceLengthStoch))
print("")
print("mean(meanPoreWidthCirc) = ", np.mean(meanPoreWidthCirc))
print("mean(meanPoreWidthStoch) = ", np.mean(meanPoreWidthStoch))
print("")
print("std(meanPoreWidthCirc) = ", np.std(meanPoreWidthCirc))
print("std(meanPoreWidthStoch) = ", np.std(meanPoreWidthStoch))
print("")
"""


#porosityMean      = np.mean(porosity, axis=1);        porosityStd      = np.std(porosity, axis=1)
#surfaceLengthMean = np.mean(surfaceLength, axis=1);   surfaceLengthStd = np.std(surfaceLength, axis=1)
#meanPoreWidthMean = np.mean(meanPoreWidth, axis=1);   meanPoreWidthStd = np.std(meanPoreWidth, axis=1)




