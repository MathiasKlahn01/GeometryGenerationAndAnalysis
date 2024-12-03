# Check the computation of the porosity
if testPorosity:
    print("************************************************************************")
    print("* Testing the computation of the porosity!                             *")
    print("************************************************************************")
    print("")

    somethingWrong = False
    for domainIndex in range(Ntest):
        Domain, porosityExact, surfaceLengthExact, meanPoreWidthExact = generate_test_domain(domainIndex)
        porosityComputed = compute_porosity(Domain)

        absDiff = np.abs(porosityComputed - porosityExact)
        if (absDiff > 1E-6):
            somethingWrong = True

        print("Domain ", domainIndex)
        print("Difference between exact and computed porosity: ", absDiff)
        print("")

    if (not somethingWrong):
        print("CONCLUSION: porosity computation is OK!")
    else:
        print("CONCLUSION: porosity computation is WRONG!")
    print("")




# Check the computation of the surface length
if testSurfaceLength:
    print("************************************************************************")
    print("* Testing the computation of the surface length!                       *")
    print("************************************************************************")
    print("")

    somethingWrong = False
    for domainIndex in range(Ntest):
        Domain, porosityExact, surfaceLengthExact, meanPoreWidthExact = generate_test_domain(domainIndex)
        surfaceLengthComputed = compute_surface_length(Domain)

        absDiff = np.abs(surfaceLengthComputed - surfaceLengthExact)
        if (absDiff > 1E-6):
            somethingWrong = True

        print("Domain ", domainIndex)
        print("Difference between exact and computed surface length: ", absDiff)
        print("")

    if (not somethingWrong):
        print("CONCLUSION: surface length computation is OK!")
    else:
        print("CONCLUSION: surface length computation is WRONG!")
    print("")




# Check the computation of the mean pore width
if testMeanPoreWidth:
    print("************************************************************************")
    print("* Testing the computation of the mean pore width!                      *")
    print("************************************************************************")
    print("")

    somethingWrong = False
    for domainIndex in range(Ntest):
        Domain, porosityExact, surfaceLengthExact, meanPoreWidthExact = generate_test_domain(domainIndex)
        meanPoreWidthComputed = compute_mean_pore_width(Domain)

        absDiff = np.abs(meanPoreWidthComputed - meanPoreWidthExact)
        if (absDiff > 1E-6):
            somethingWrong = True

        print("Domain ", domainIndex)
        print("Difference between exact and computed mean pore width: ", absDiff)
        print("")

    if (not somethingWrong):
        print("CONCLUSION: mean pore width computation is OK!")
    else:
        print("CONCLUSION: mean pore width computation is WRONG!")
    print("")
    print("")