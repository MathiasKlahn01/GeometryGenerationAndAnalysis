# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from   GeometryFunctions import *


# Decide which tests should be performed
testTestDomains = True
Ntest = 2

testDomainGeneration = True


# Check that the properties of the test domains are computed correctly
if testTestDomains:
    print("")
    print("************************************************************************")
    print("* Testing the properties of the test domains                           *")
    print("************************************************************************")
    print("")

    errorPorosity      = False
    errorSurfaceLength = False
    errorMeanPoreWidth = False
    for domainIndex in range(Ntest):
        Domain, porosityExact, surfaceLengthExact, meanPoreWidthExact = generate_circular_test_domain(domainIndex)
        
        # Porosity
        porosityComputed      = compute_porosity(Domain)
        if (np.abs(porosityComputed - porosityExact) > 1E-6):
            errorPorosity = True

        # Surface length
        surfaceLengthComputed = compute_surface_length(Domain)
        if (np.abs(surfaceLengthComputed - surfaceLengthExact) > 1E-6):
            errorSurfaceLength = True

        # Mean pore width
        meanPoreWidthComputed = compute_mean_pore_width(Domain)
        if (np.abs(meanPoreWidthComputed - meanPoreWidthExact) > 1E-6):
            errorMeanPoreWidth = True

    
    print("CONCLUSION:")
    if (not errorPorosity):
        print("1) Porosity: OK!")
    else: 
        print("1) Porosity: something wrong!")
    
    if (not errorSurfaceLength):
        print("2) SurfaceLength: OK!")
    else: 
        print("2) SurfaceLength: something wrong!")
    
    if (not errorMeanPoreWidth):
        print("3) Mean pore width: OK!")
    else: 
        print("3) Mean pore width: something wrong!")
    
print("")




# Check the computation of circular obstacle domains
NxList      = np.array([1000])
NyList      = np.array([1000])
rList       = np.array([  20.0])
minDistList = np.array([  10.0])
porList     = np.array([   0.5])
NtryList    = np.array([100000])
if testDomainGeneration:
    print("************************************************************************")
    print("* Testing the generation of circular obstacle domains!                 *")
    print("************************************************************************")
    print("")

    somethingWrong = False
    for n in range(NxList.size):
        print("Generating domain", n)
        Domain = generate_circular_domain(NxList[n], NyList[n], rList[n], minDistList[n], porList[n], NtryList[n], 1)
        #vertices, elements = generate_structured_mesh(Domain)
        #print("The generation required", nIter, "iterations")

        plt.figure(n)
        plt.contourf(Domain)
        plt.savefig("Figures/CircularDomainNx" + str(NxList[n]) + "Ny" + str(NyList[n]) + "r" + str(rList[n]) + "minDist" + str(minDistList[n]) + ".pdf")

        porosityComputed      = compute_porosity(Domain);          print("Porosity = ", porosityComputed)
        surfaceLengthComputed = compute_surface_length(Domain);    print("surface length = ", surfaceLengthComputed)
        meanPoreWidthComputed = compute_mean_pore_width(Domain);   print("Mean pore width = ", meanPoreWidthComputed)
        print("")
        print("")

    if (not somethingWrong):
        print("CONCLUSION: Domain and mesh generation is OK!")
    else:
        print("CONCLUSION: Domain and mesh generation is WRONG!")
    print("")
    print("")




