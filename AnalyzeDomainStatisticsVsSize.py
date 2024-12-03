# Import libraries
import numpy             as np
import matplotlib.pyplot as plt
import os


# Define parameters used for both types of domain
NStat    = 100
sizeMin  = 100
sizeMax  = 1000
sizeStep = 100
sizeList = np.arange(sizeMin, sizeMax+sizeStep, sizeStep)

# Define parameters related to the circular obstacle domains
r       = 20.0   # r = 20.0 corrsponds to h = r/20, i.e. h_nondim = 0.05
minDist = 2.0    # distMin = 2.0 corresponds to distMin = r/10, i.e. distMin_nondim = 0.1 (at least when r = 20.0...)
por     = 0.5
NTry    = 1000

# Define parameters related to the stochastic domains
gamma    = 0.5
lambdaxy = 20.0



# Load data from file
dataPath = 'Figures/Data/'

porosityCirc      = np.load(dataPath + 'porosityCircNstat' + str(NStat) + 'sizeMin' + str(sizeMin) + 'sizeMax' + str(sizeMax) + 'sizeStep' + str(sizeStep) + \
                             'r' + str(r) + 'minDist' + str(minDist) + 'por' + str(por) + 'NTry' + str(NTry) + '.npy')
surfaceLengthCirc = np.load(dataPath + 'surfaceLengthCircNstat' + str(NStat) + 'sizeMin' + str(sizeMin) + 'sizeMax' + str(sizeMax) + 'sizeStep' + str(sizeStep) + \
                             'r' + str(r) + 'minDist' + str(minDist) + 'por' + str(por) + 'NTry' + str(NTry) + '.npy')
meanPoreWidthCirc = np.load(dataPath + 'meanPoreWidthCircNstat' + str(NStat) + 'sizeMin' + str(sizeMin) + 'sizeMax' + str(sizeMax) + 'sizeStep' + str(sizeStep) + \
                             'r' + str(r) + 'minDist' + str(minDist) + 'por' + str(por) + 'NTry' + str(NTry) + '.npy')


porosityStoch      = np.load(dataPath + 'porosityStochNstat' + str(NStat) + 'sizeMin' + str(sizeMin) + 'sizeMax' + str(sizeMax) + 'sizeStep' + str(sizeStep) + \
                             'gamma' + str(gamma) + 'lambdaxy' + str(lambdaxy) + '.npy')
surfaceLengthStoch = np.load(dataPath + 'surfaceLengthStochNstat' + str(NStat) + 'sizeMin' + str(sizeMin) + 'sizeMax' + str(sizeMax) + 'sizeStep' + str(sizeStep) + \
                             'gamma' + str(gamma) + 'lambdaxy' + str(lambdaxy) + '.npy')
meanPoreWidthStoch = np.load(dataPath + 'meanPoreWidthStochNstat' + str(NStat) + 'sizeMin' + str(sizeMin) + 'sizeMax' + str(sizeMax) + 'sizeStep' + str(sizeStep) + \
                             'gamma' + str(gamma) + 'lambdaxy' + str(lambdaxy) + '.npy')


# Write results to terminal
print("std(Porosity) (circ): ", np.std(porosityCirc, axis=1))
print("")
print("std(Porosity) (stoch): ", np.std(porosityStoch, axis=1))
print("")
print("")
print("std(surface length) (circ): ", np.std(surfaceLengthCirc, axis=1))
print("")
print("std(surface length) (stoch): ", np.std(surfaceLengthStoch, axis=1))
print("")
print("")
print("std(Mean pore width) (circ): ", np.std(meanPoreWidthCirc, axis=1))
print("")
print("std(Mean pore width) (stoch): ", np.std(meanPoreWidthStoch, axis=1))
print("")
print("")



# Plot the data
plt.figure(1)
plt.plot(sizeList, np.mean(porosityCirc, axis=1), marker='o', linestyle='-', color='red')
plt.plot(sizeList, np.mean(porosityCirc, axis=1)-np.std(porosityCirc, axis=1), linestyle='--', color='red')
plt.plot(sizeList, np.mean(porosityCirc, axis=1)+np.std(porosityCirc, axis=1), linestyle='--', color='red')

plt.plot(sizeList, np.mean(porosityStoch, axis=1), marker='o', linestyle='-', color='black')
plt.plot(sizeList, np.mean(porosityStoch, axis=1)-np.std(porosityStoch, axis=1), linestyle='--', color='black')
plt.plot(sizeList, np.mean(porosityStoch, axis=1)+np.std(porosityStoch, axis=1), linestyle='--', color='black')

plt.xlabel(r'$L_x/h$', fontsize=12)
plt.ylabel(r'$\langle$Porosity$\rangle$', fontsize=12)
plt.title(r'$N_{stat} = $' + str(NStat) + r', $r/h = $' + str(r) + r', $d_{min}/h = $' + str(minDist) + r', $\Psi = $' + str(por) + r', $N_{try} = $' + str(NTry) + ', \n' \
          r'$\gamma = $' + str(gamma) + r', $\lambda = $' + str(lambdaxy), fontsize=10)

plt.legend(handles=[plt.Line2D([0], [0], color='red', label=r'Circular domain ($\pm \sigma$)'), 
                    plt.Line2D([0], [0], color='black', label=r'Stochastic domain ($\pm \sigma$)')])

plt.savefig('Figures/meanPorosityNStat' + str(NStat) + 'sizeMin' + str(sizeMin) + 'sizeMax' + str(sizeMax) + 'sizeStep' + str(sizeStep) + \
                            'r' + str(r) + 'minDist' + str(minDist) + 'por' + str(por) + 'NTry' + str(NTry) + \
                            'gamma' + str(gamma) + 'lambdaxy' + str(lambdaxy) + '.pdf')



plt.figure(2)
plt.plot(sizeList, np.mean(surfaceLengthCirc, axis=1), marker='o', linestyle='-', color='red')
plt.plot(sizeList, np.mean(surfaceLengthCirc, axis=1)-np.std(surfaceLengthCirc, axis=1), linestyle='--', color='red')
plt.plot(sizeList, np.mean(surfaceLengthCirc, axis=1)+np.std(surfaceLengthCirc, axis=1), linestyle='--', color='red')

plt.plot(sizeList, np.mean(surfaceLengthStoch, axis=1), marker='o', linestyle='-', color='black')
plt.plot(sizeList, np.mean(surfaceLengthStoch, axis=1)-np.std(surfaceLengthStoch, axis=1), linestyle='--', color='black')
plt.plot(sizeList, np.mean(surfaceLengthStoch, axis=1)+np.std(surfaceLengthStoch, axis=1), linestyle='--', color='black')

plt.xlabel(r'$L_x/h$', fontsize=12)
plt.ylabel(r'$\langle$Normalized surface length$\rangle /h$', fontsize=12)
plt.title(r'$N_{stat} = $' + str(NStat) + r', $r/h = $' + str(r) + r', $d_{min}/h = $' + str(minDist) + r', $\Psi = $' + str(por) + r', $N_{try} = $' + str(NTry) + ', \n' \
          r'$\gamma = $' + str(gamma) + r', $\lambda = $' + str(lambdaxy), fontsize=10)

plt.legend(handles=[plt.Line2D([0], [0], color='red', label=r'Circular domain ($\pm \sigma$)'), 
                    plt.Line2D([0], [0], color='black', label=r'Stochastic domain ($\pm \sigma$)')])

plt.savefig('Figures/meanSurfaceLengthNStat' + str(NStat) + 'sizeMin' + str(sizeMin) + 'sizeMax' + str(sizeMax) + 'sizeStep' + str(sizeStep) + \
                            'r' + str(r) + 'minDist' + str(minDist) + 'por' + str(por) + 'NTry' + str(NTry) + \
                            'gamma' + str(gamma) + 'lambdaxy' + str(lambdaxy) + '.pdf')



plt.figure(3)
plt.plot(sizeList, np.mean(meanPoreWidthCirc, axis=1), marker='o', linestyle='-', color='red')
plt.plot(sizeList, np.mean(meanPoreWidthCirc, axis=1)-np.std(meanPoreWidthCirc, axis=1), linestyle='--', color='red')
plt.plot(sizeList, np.mean(meanPoreWidthCirc, axis=1)+np.std(meanPoreWidthCirc, axis=1), linestyle='--', color='red')

plt.plot(sizeList, np.mean(meanPoreWidthStoch, axis=1), marker='o', linestyle='-', color='black')
plt.plot(sizeList, np.mean(meanPoreWidthStoch, axis=1)-np.std(meanPoreWidthStoch, axis=1), linestyle='--', color='black')
plt.plot(sizeList, np.mean(meanPoreWidthStoch, axis=1)+np.std(meanPoreWidthStoch, axis=1), linestyle='--', color='black')

plt.xlabel(r'$L_x/h$', fontsize=12)
plt.ylabel(r'$\langle$Mean pore width$\rangle$', fontsize=12)
plt.title(r'$N_{stat} = $' + str(NStat) + r', $r/h = $' + str(r) + r', $d_{min}/h = $' + str(minDist) + r', $\Psi = $' + str(por) + r', $N_{try} = $' + str(NTry) + ', \n' \
          r'$\gamma = $' + str(gamma) + r', $\lambda = $' + str(lambdaxy), fontsize=10)

plt.legend(handles=[plt.Line2D([0], [0], color='red', label=r'Circular domain ($\pm \sigma$)'), 
                    plt.Line2D([0], [0], color='black', label=r'Stochastic domain ($\pm \sigma$)')])

plt.savefig('Figures/meanMeanPoreLengthNStat' + str(NStat) + 'sizeMin' + str(sizeMin) + 'sizeMax' + str(sizeMax) + 'sizeStep' + str(sizeStep) + \
                            'r' + str(r) + 'minDist' + str(minDist) + 'por' + str(por) + 'NTry' + str(NTry) + \
                            'gamma' + str(gamma) + 'lambdaxy' + str(lambdaxy) + '.pdf')

