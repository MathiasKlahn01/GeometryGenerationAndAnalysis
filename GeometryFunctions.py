#===============================================================================================#
# Import libraries                                                                              #
#===============================================================================================#

import numpy as np
from   scipy.ndimage import label


#===============================================================================================#
# Functions related to stochastic geometry generation based on Hyman & Winter ("Stochastic gen- #
# eration of explicit pore structures by thresholding Gaussian random fields", 2014)            #
#===============================================================================================#

def generate_stochastic_test_domain(domainIndex):
    # Note: 1 denotes fluid, 0 denotes solid!
    if (domainIndex == 0):
        Domain = np.array([[1, 0, 1, 1, 0, 1, 0, 0, 1],
                           [1, 1, 1, 1, 0, 0, 0, 0, 1],
                           [1, 0, 0, 1, 0, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 1],
                           [1, 1, 0, 1, 0, 0, 0, 1, 1],
                           [1, 1, 1, 1, 1, 1, 0, 0, 1],
                           [1, 0, 1, 1, 0, 0, 0, 1, 1]], dtype=np.uint8)
        porosity      = 0.650793650794
        surfaceLength = 0.380952380952   #48.00000000000
        meanPoreWidth = 2.157894736842
    elif (domainIndex == 1):
        Domain = np.array([[1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                           [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                           [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                           [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
                           [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
                           [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                           [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1]], dtype=np.uint8)
        porosity      = 0.666666666666
        surfaceLength = 0.388888888888   #98.00000000000
        meanPoreWidth = 2.270270270270
    elif (domainIndex == 2):
        Domain = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1],
                           [1, 0, 1, 1, 1, 1, 1, 0, 1],
                           [1, 0, 0, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 1, 1, 1, 1, 0, 1],
                           [1, 0, 1, 0, 0, 0, 1, 0, 1],
                           [1, 0, 1, 1, 1, 1, 1, 0, 1]], dtype=np.uint8)
        porosity      = 0.648148148148
        surfaceLength = 0.361111111111   #39.00000000000
        meanPoreWidth = 2.058823529412
    elif (domainIndex == 3):
        Domain =  np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                            [1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1]])
        porosity      = 0.590000000000
        surfaceLength = 0.225000000000   #45.00000000000
        meanPoreWidth = 3.277777777777
    
    # Return the result of the computation
    return Domain, porosity, surfaceLength, meanPoreWidth



def compute_periodic_labels(Domain):
    labels, numberOfFeatures = label(Domain)

    for y in range(labels.shape[0]):
        lL = labels[y, 0]
        lR = labels[y, -1]
        if lL > 0 and lR > 0:
            labels[labels == max(lL, lR)] = min(lL, lR)

    for x in range(labels.shape[1]):
        lL = labels[0, x]
        lR = labels[-1, x]
        if lL > 0 and lR > 0:
            labels[labels == max(lL, lR)] = min(lL, lR)
    return labels


def check_periodicity(Domain):
    Domain2x = np.hstack([Domain, Domain])
    is_periodic_x = compute_periodic_labels(Domain2x).max() == 1

    Domain2y = np.vstack([Domain, Domain])
    is_periodic_y = compute_periodic_labels(Domain2y).max() == 1

    # Return the result of the computation
    return is_periodic_x and is_periodic_y


def generate_stochastic_domain(Nx, Ny, gamma, lambdaxy, rngSeed):
    # Note: 1 denotes fluid, 0 denotes solid!

    # Initialize the random number generator
    np.random.seed(rngSeed)

    # Evaluate the Gaussian kernel function
    x0 = -0.5*Nx;   xGrid = np.linspace(x0, x0+(Nx-1), Nx)
    y0 = -0.5*Ny;   yGrid = np.linspace(y0, y0+(Ny-1), Ny)

    norm  = 1.0/(2.0*np.pi*np.sqrt(lambdaxy*lambdaxy))
    xexp  = np.exp(-0.5*np.square(xGrid)/lambdaxy)
    yexp  = np.exp(-0.5*np.square(yGrid)/lambdaxy)
    K     = norm*np.outer(xexp, yexp)
    Kfft2 = np.fft.fft2(K)

    # Construct a periodic domain
    periodic = False
    nIter = 1
    while (not periodic):
        nIter = nIter + 1

        # Evaluate the random uniformly distributed field, convolve it with the Gaussian kernel and normalize the result
        u = np.random.rand(Nx, Ny)
        T = np.real(np.fft.ifft2(Kfft2*np.fft.fft2(u)))

        TMin = np.min(T)
        TMax = np.max(T)
        T    = (T - TMin)/(TMax - TMin)

        # Compute the domain matrix (D[n,m] = 0 if the (n,m)th grid point is part of the pore space and D[n,m] = 1 if the (n,m)th grid point is part of the solid space)
        Domain = (T < gamma).astype(np.uint8)

        # Find the connected components of the domain and check if it is periodic
        labels = compute_periodic_labels(Domain)
        ic_max = np.argmax([int(np.sum(labels == ic)) for ic in range(1, labels.max()+1)]) + 1

        Domain = labels == ic_max
        periodic = check_periodicity(Domain)

    # Return the result of the computation
    return Domain, nIter


def generate_structured_mesh(Domain):
    # Extract the number of grid points in the domain
    Ny, Nx = Domain.shape

    # Compute the list of vertices
    vertices = []
    for ny in range(Ny+1):
        for nx in range(Nx+1):
            vertices.append((1.0*nx, 1.0*ny))
    vertices = np.array(vertices)

    # Compute the list of elements
    elements = []
    for ny in range(Ny):
        for nx in range(Nx):
            if (Domain[ny, nx] == 1):
                I0 = nx + ny*(Nx+1)
                I1 = (nx+1) + ny*(Nx+1)
                I2 = (nx+1) + (ny+1)*(Nx+1)
                I3 = nx + (ny+1)*(Nx+1)
                elements.append((I0, I1, I3))
                elements.append((I1, I2, I3))
    elements = np.array(elements)

    # Remove vertices that are not used by any element
    unique_vertices = np.unique(elements)
    oldToNew = np.zeros(np.max(unique_vertices)+1, dtype=int)
    oldToNew[unique_vertices] = np.arange(len(unique_vertices))

    vertices = vertices[unique_vertices]
    elements = oldToNew[elements]

    # Return the result of the computation
    return vertices, elements



#===============================================================================================#
# Functions related to geometry generation based on randomly places circular obstacles (as done #
# in BERNAISE)                                                                                  #
#===============================================================================================#

def generate_circular_test_domain(domainIndex):
    # Note: 1 denotes fluid, 0 denotes solid!
    if (domainIndex == 0):
        Domain = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                           [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                           [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                           [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                           [1, 1, 0, 0, 0, 0, 0, 0, 1, 1], 
                           [1, 1, 1, 0, 0, 0, 0, 1, 1, 1], 
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
        porosity      = 0.680000000000
        surfaceLength = 0.120000000000
        meanPoreWidth = 4.250000000000
    if (domainIndex == 1):
        Domain = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                           [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                           [1, 0, 0, 0, 0, 1, 1, 0, 0, 0 ,0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                           [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], 
                           [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1], 
                           [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
        porosity      = 0.604166666666
        surfaceLength = 0.166666666666
        meanPoreWidth = 4.142857142857    

    # Return the result of the computation 
    return Domain, porosity, surfaceLength, meanPoreWidth


def generate_random_point(Nx, Ny):
    x = np.random.uniform(0, Nx)
    y = np.random.uniform(0, Ny)

    # Return the result of the computation
    return (x,y)


def point_to_point_squared_distance(point1, point2):
    dist2 = np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1])

    # Return the result of the computation
    return dist2





def generate_circular_domain(Nx, Ny, r, minDist, por, Ntry, rngSeed):
        # Initialize the random number generator
    np.random.seed(rngSeed)
    
    # Compute the target number of obstacles
    NobsTarget = int(np.ceil((1.0-por)*Nx*Ny/(np.pi*r*r)))

    # Attempt to generate NobsTarget cylindrical obstacles
    obstacleCenters = []
    domainIsFull    = False
    dSquared        = np.square(2.0*r + minDist)
    while (not domainIsFull) and (len(obstacleCenters) < NobsTarget):
        domainIsFull = True
        for n in range(Ntry):
            newPoint = generate_random_point(Nx, Ny)
            newPointIsTooClose = False
            for m in range(len(obstacleCenters)):
                if (point_to_point_squared_distance(newPoint, obstacleCenters[m]) < dSquared):
                    newPointIsTooClose = True
                    break 
            
            if (not newPointIsTooClose):
                obstacleCenters.append(newPoint)
                domainIsFull = False
                break

    print("Target number of obstacles = ", NobsTarget)
    print("Actual number of obstacles = ", len(obstacleCenters))

    # Fill in the domain
    Domain = np.ones((Ny,Nx), dtype=np.uint8)
    r2     = np.square(r) 
    for point in obstacleCenters:
        nxMin = int(max(0.0, np.floor(point[0]-1.01*r)));   nxMax = int(min(Nx, np.ceil(point[0]+1.01*r)))
        nyMin = int(max(0.0, np.floor(point[1]-1.01*r)));   nyMax = int(min(Ny, np.ceil(point[1]+1.01*r)))

        for ny in range(nyMin, nyMax):
            for nx in range(nxMin, nxMax):
                dist2 = np.square(point[0] - (nx+0.5)) + np.square(point[1] - (ny+0.5))
                if (dist2 < r2):
                    Domain[ny,nx] = 0


    # Return the result of the computation
    return Domain




    print("Nobs = ", Nobs)



#===============================================================================================#
# Utility functions that can be used for both types of domain                                   #
#===============================================================================================#

def compute_porosity(Domain): 
    # Compute the porosity
    Ny, Nx   = Domain.shape
    porosity = np.sum(Domain)/(Nx*Ny)

    # Return the result of the computation
    return porosity


def compute_surface_length(Domain):
    # Notes:
    # 1) The current version of this function does NOT consider periodicity!
    # 2) The is dimensionless

    # Compute the surface length
    Ny, Nx = Domain.shape
    surfaceLength = 0.0
    for ny in range(Ny):
        for nx in range(Nx-1):
            if (Domain[ny,nx] != Domain[ny,nx+1]):
                surfaceLength = surfaceLength + 1.0

    for nx in range(Nx):
        for ny in range(Ny-1):
            if (Domain[ny,nx] != Domain[ny+1,nx]):
                surfaceLength = surfaceLength + 1.0

    surfaceLength = surfaceLength/(2.0*Nx*Ny)

    # Return the result of the computation
    return surfaceLength




def compute_mean_pore_width(Domain): 
    # Notes: 
    # 1) The current version of this function does NOT consider periodicity!
    # 2) The result is dimensionless and must be scaled with the mesh size

    # Compute the mean pore width
    Ny, Nx = Domain.shape
    poreWidthList = []
    for ny in range(Ny):
        nx = 0
        while (nx < Nx):
            while (Domain[ny,nx] == 0):
                if (nx < Nx-1):
                    nx = nx+1
                if (nx == Nx-1):
                    break

            if (nx == Nx-1) and (Domain[ny,nx] == 0):
                break

            poreWidth = 0
            while (Domain[ny,nx] == 1):
                if (nx < Nx):
                    poreWidth = poreWidth + 1
                    nx = nx+1
                if (nx == Nx):
                    break

            poreWidthList.append(poreWidth)

    poreWidthList = np.array(poreWidthList)
    meanPoreWidth = np.mean(poreWidthList)
    
    # Return the result of the computation
    return meanPoreWidth