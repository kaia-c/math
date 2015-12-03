# Calculates the linearA regression model and plots the data
# Limitations: only returns the most basic regression outputs

from scipy import stats
import numpy as np
import pylab

###normalityTests
##def normalityTests(x,y,a):
##    ok=True
##   
##    #test for normalA distribution
##    nx=x.size
##    ny=y.size
##    if nx>=20:
##        print ('Sample 1: normaltest teststat = %6.6f pvalue = %6.4f' % stats.normaltest(x))
##    elif nx >=8:
##        print ('Sample 1: normalA skewtest teststat = %6.6f pvalue = %6.4f' % stats.skewtest(x))
##        print('Sample 1: median = %6.6f mean = %6.6f ' % (stats.nanmedian(x), stats.nanmean(x)))
##    else:
##        print('Sample 1: median = %6.6f mean = %6.6f ' % (stats.nanmedian(x), stats.nanmean(x)))
##    if nx>=20:
##        print ('Sample 2: normaltest teststat = %6.6f pvalue = %6.4f' % stats.normaltest(y))
##    elif nx >=8:
##        print ('Sample 2: normalA skewtest teststat = %6.6f pvalue = %6.4f' % stats.skewtest(y))
##        print('Sample 2: median = %6.6f mean = %6.6f ' % (stats.nanmedian(y), stats.nanmean(y)))
##    else:
##        print('Sample 2: median = %6.6f mean = %6.6f ' % (stats.nanmedian(y), stats.nanmean(y)))
##
##
##    #if p value lower then a, reject null hypothesis of norm distribution
##    # % stats.ttest_1samp(x, y)
##
##    conXY = np.concatenate([x,y])
##    nXY = conXY.size
##    if nXY >= 8:
##        if nXY >=20:
##            zXY, pXY =  stats.normaltest(conXY)
##        else:
##            zXY, pXY = stats.skewtest(conXY)
##            print('Due to small size of n, can only look at skew.')
##        if pXY < a:
##            print('Likely NOT normally distributed data in each set. \nCombined z = %6.6f  p-value = %6.6f' %(zXY, pXY))
##            if pXY > a:
##                print('Combined data also NOT normally distributes within %2.2f level of significance.'%(a))
##                ok=False
##            else:
##                print('Combined data IS normally distributes within %2.2f level of significance.'%(a))
##        else:
##            print('Likely normally distributed data independently. \nCombined z = %6.6f  p-value = %6.6f' %(zXY, pXY))
##    else:
##        altTest = input('combined n to small to evaluate normality effectivly.\n median = %6.6f mean = %6.6f.\nAre these sufficently close? ' % (stats.nanmedian(conXY), stats.nanmean(conXY)))
##        if altTest.upper() == 'NO' or altTest.upper() == 'N':
##            of=False
##    return ok
##########################END def normalityTests(x,y):


def ssd(x,y):
    sumSquares= np.sum((x-y)**2)
    return sumSquares

def predictY(intercept, slope, x):
    return intercept + slope * x

def getErrors(intercept, slope,  x, y):
    e = np.array([])
    for xi, yi, in zip(x,y):
        e=np.append(e,(yi-predictY(intercept, slope,  xi)))
    return e

def standardize(x,mean, stdev):
    return((x-mean)/stdev)

def normProbPlot(e):
    e=np.sort(e)
    ze=np.array([])
    for i in range(1, e.size+1):
        ze=np.append(ze, stats.norm.ppf((i/(e.size+1)),0,1))
    #for i,j in zip(e,ze):
    #    print('%f %f'%(i,j))
    pylab.plot(ze, e, 'o')
    pylab.xlabel('Z values')
    pylab.ylabel('Residuals')
    pylab.title('Normal Probability of Errors')
    pylab.show()

def plotErrors(x,e):
    pylab.plot(x, e, 'o')
    pylab.xlabel('X')
    pylab.ylabel('Residuals')
    pylab.title('Residual Plot')
    pylab.show()
    
def assumptions(intercept, slope,  x, y, a):
    normal = True
    equal=True
    linear=True
    indep=True
    ok=True
    print('[L]inearity of origional data')
    print('[I]ndependence of errors')
    print('[N]ormality of errors')
    print('[E]qual variance of errors')
    e = getErrors(intercept, slope,  x, y)
    print('If their is NO apparant pattern, the linear model IS appropriate.')
    print('If y apprears to be more frequently higher/lower dependent on size of X, variances are NOT equal.')
    plotErrors(x,e)
    linearA = input('Is there NO apparant pattern? [Y]es or [N]o: ')
    if linearA.upper() == 'NO' or linearA.upper() == 'N':
        linear=False
    equalA = input('Is there a random distribution without x influincing Y?\nNO linear pattern? [Y]es or [N]o: ')
    if equalA.upper() == 'NO' or equalA.upper()=='N':
        equal=False
    print('If data is normal, it will plot on a straight line.')
    normProbPlot(e)
    normalA=input('Appx straight line/ normal distribution? [Y]es or [N]o: ')
    if normalA.upper() == 'NO' or normalA.upper() == 'N':
        normal=False
    indepA=input('Was data gathered for both sets independently? [Y]es or [N]o: ')
    if indepA.upper()=='NO' or indepA.upper()=='N':
        indep=False
    if normal and indep and equal and linear:
        ok = True
    else:
        ok=False
    if ok:
        print('Assuming data was independently gathered, assumptions of regressions are met.\nlinear model appropriate.')
    else:
        print('Based on your observations, assumtion of regression has been violated.')
        if not normal:
            print('Errors are not normally distributed.')
        if not equal:
            print('Errors are not equal')
        if not linear:
            print('Origional data is not linear')
        if not indep:
            print('Errors not independent')
##    print('Results from the numbers regarding normality as folows:\n')
##    normal=normalityTests(x,y, a)
##    if normal and indep and equal and linear:
##        ok = True
##    else:
##        ok=False
##    print('\nAfter looking at the numbers, it appears that it is ' + str(ok) + ' that this data fits the assumptions of regression\n')
    return ok
##################END def assumptions(intercept, slope,  x, y, a):

def tTest(slope, slope_std_error, degFreedom, a):
    #get t critical values - 
    #Studnt, n=999, p<0.05, 2-tail
    #equivalent to Excel TINV(0.05,999)
    #print stats.t.ppf(1-0.025, 999)
    #Studnt, n=999, p<0.05%, Single tail
    #equivalent to Excel TINV(2*0.05,999)
    #print stats.t.ppf(1-0.05, 999)
    #http://stackoverflow.com/questions/19339305/python-function-to-get-the-t-statistic
    nullHyp=0
    tStat=(slope+nullHyp)/slope_std_error
    tails = 2
    if tails == 1:
        tCrit = stats.t.ppf(1-a, degFreedom)
    elif tails == 2:
        tCrit = stats.t.ppf(1-(a/2), degFreedom)
    halfWidth = slope_std_error*tCrit
    lowerConfidence=slope-halfWidth
    upperConfidence = slope+halfWidth
    print('Are the ctitical values of t: %6.6f < tStat= %6.6f < %6.6f?'%((-1*tCrit),tStat, tCrit))
    if tStat > tCrit or tStat < (-1*tCrit):
        print('You CAN reject the Null hypothesis as their IS a linear relationship between x and y. ')
    else:
        print('You CANNOT reject the Null hypothesis as their is NOT a linear relationship between x and y. ')
    print('With ' + str(100*(1-a)) + ' percent confidence, you population slope is in the inteval:')
    print('\nCalc: interval half width = %6.6f \nConfidence Interval\n\t%6.6f < slope=%6.6f < %6.6f?\n'  %(halfWidth, lowerConfidence, slope, upperConfidence))
    
#####################end def tTest(slope, slope_std_error, degFreedom):


def sumSquares(x,y, intercept, slope):
    #Sum Square Error, Regression, and Total
    SY = 0.0
    SYSquared = 0.0
    SXY= 0.0
    SSR=0.0

    for yi in y:
        SY+=yi
    for yi in y:
        SYSquared+=(yi**2)
    for yi, xi in zip(y,x):
        SXY+=(xi*yi)

    SST=SYSquared - SY**2/x.size
    SSR = (intercept*SY)+(slope*SXY)-((SY**2)/x.size)
    #print('intercept= %6.6f*SY= %6.6f)+(slope= %6.6f*SXY= %6.6f)-((SY= %6.6f**2)/x.size= %6.6f'%(intercept,SY,slope,SXY,SY,x.size))
    SSE = SST - SSR
    rSquared = SSR/SST
    adjustedR = 1-((x.size-1)/(x.size-2))*(SSE/SST)
    print('\nSSR = %6.6f\nSSE = %6.6f\nSST = %6.6f\nr-squared = %6.6f\nadjusted r-squared = %6.6f' %(SSR, SSE, SST, rSquared, adjustedR))
    print('R-Squared = the proportion of the variation in Y explained by variability in X.\nThe regression model is %6.2f-%6.2f percent useful for predicting Y.'%((adjustedR*100),(rSquared*100)))
    #standard error of the estimate
    SYX = (SSE/(x.size-2))**(1/2)
    print('standard error of the estimate (SYX) = %6.6f'%SYX)
    return(SSR, SSE, SST, rSquared, SYX)
##################end def sumSquares(x,y, intercept, slope)

##def simpleLinearRegression(x,y):
##    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
##    #predict_y = predictY(intercept, slope,  x)
##
##    #plotRegression=input('Plot the linear regression model? [Y]es or [N]o: ')
##    #if plotRegression.upper()=='Y' or plotRegression.upper()=='YES':
####        # Plotting
####        pylab.plot(x, y, 'o')
####        pylab.plot(x, predictY(intercept, slope,  x), 'k-')
####        pylab.title('Data vs. Prediction')
####        pylab.xlabel('x')
####        pylab.ylabel('y')
####        pylab.show()
##        
##    #print('r_value = %6.6f\np_value = %6.6f\nslope_std_error = %6.6f\nresidual_std_error = %6.6f'%(r_value, p_value, slope_std_error, residual_std_error))
##    
##    return (intercept, slope, slope_std_error)
#####end regression()#####################

def durbinWatson(e):
    numerator=0.0
    denominator=0.0
    for i in range(1, e.size):
        numerator+=(e[i]-e[i-1])**2
    for i in e:
        denominator+=i**2
    return(numerator/denominator)
#################END def durbinWatson(e)


def autoCorr(x,e):
    print('if there is no pattern, no autocorrelation.\nIf there is a linear pattern, positive autocorrelation.\nIf there is an back-and-forth high-low cpattern, negative autocorrelation.')
    plotErrors(x,e)
    autoA=int(input('Is there:\n1)\tno pattern/autocorrelation\n2)\tLinear pattern-positive correlation\n3)\tup-down pattern-negative correlation?\n'))
    if autoA==1:
        print('no autocorrelation')
    elif autoA==2:
        print('positive autocorrelation')
    elif autoA==3:
        print('negative autocorrelation')

    d=durbinWatson(e)
    print('Durbin Waton D-Stat = %6.6f'%d)
    if d>=3:
        print('negative autocorrelation')
    elif d<=1:
        print('positive autocorrelation')
    elif d>1.5 and d<5:
        print('no autocorrelation')
    else:
        print('indeteminate autocorrelation')
#############END def autoCorr(x,e)


def fTest(SSR, SSE, degreesFreedom, a):
    fStat=SSR/(SSE/degreesFreedom)
    print('\nfStat = %6.6f' % fStat)
    fCrit = stats.f.cdf(fStat, degreesFreedom, degreesFreedom)
    print('f critical value = %6.6f' % fCrit)
    print('It is ' + str(fCrit>a)+ ' that these values have equal variences and are vadid, and you should reject null hypothesis.')
    return (fCrit>a)
    
#####################end def fTest(SSR, SSE, x.size):
   
######MAIN()##################
# Fit the model



run= True
while run:
    SSR=-1
    SSE=-1
    SYX=-1
    ok=True
    probablyOK=True
    enterMenu=False
    data=False
    dataInput=int(input('1)\tEnter Data to obtain Stats\n2)\tEnter Statistics as Needed for calculation?\n'))
    if dataInput == 1:
        x = np.array([])
        xInput = input('\nInput x values : ')
        xInput=xInput.split()
        for i in xInput:
            x = np.append(x, float(i))
        y = np.array([])
        yInput = input('\nInput y values : ')
        yInput=yInput.split()
        for i in yInput:
            y = np.append(y, float(i))
        slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
        confidence=float(input("Confidence level? %"));
        a=1-confidence/100
        degreesFreedom = x.size-2
        enterMenu=True
        data=True
    elif dataInput == 2:
        enterMenu=True
    #menu loop
    while(enterMenu):
        print('\n_________________________________________________________________________\nDo you want to:')
        print('1)\tPlot Regression model')
        print('2)\tFind an X value based on a given Y')
        print('3)\tGet Sum of Squares')
        print('4)\tGraphically check assumptions of regression model')
        print('5)\tDo a t-test for linearity with Null Hypothesis slope=0?')
        print('6)\tDo f-test for linearity with Null Hypothesis slope=0?')
        print('7)\tCheck for autocorrelation - for data collected over time?')
        print('8)\tGet Residual Standard Error')
        print('9)\tGet the coeficient of correlation')
        print('10)\tGet confidence interval for population mean response')
        print('11)\tGet r-squared, coefficient of determination')
        print('12)\tQuit this calculation')

        menuInput=int(input('_________________________________________________________________________\n'))
        if menuInput == 1:
            if data:
                # Plotting
                pylab.plot(x, y, 'o')
                pylab.plot(x, predictY(intercept, slope,  x), 'k-')
                pylab.title('Data vs. Prediction')
                pylab.xlabel('x')
                pylab.ylabel('y')
                pylab.show()
            else:
                print('Stats need to be provided for this option.')
        if menuInput == 2:# find y based on x
            if not data:
                intercept = float(input('Enter Intercept, B0 = '))
                slope = float(input('Enter slope, B1 = '))
            userX = float(input('for b0 = %6.6f and b1 = %6.6f in y=b0 + b1x, for x = ' % (intercept, slope)))
            userY = predictY(intercept, slope,  userX)
            print('the predicted Y = %6.6f' %userY)
        elif menuInput == 3:#sum of squares
            if data:
                SSR, SSE, SST, rSquared, SYX = sumSquares(x,y, intercept, slope)
            else:
                print('Stats need to be provided for this option.')
        elif menuInput == 4:# graphical assumptions
            if data:
                probablyOK=assumptions(intercept, slope,  x, y, a)
            else:
                print('Stats need to be provided for this option.')
        elif menuInput == 5:# tTest
            if not data:
                slope = float(input('Enter slope: '))
                slope_std_error=float(input('Enter slope_std_error: '))
                degreesFreedom = float(input('Enter degreesFreedom (n - 2): '))
                a = float(input('Enter alpha (confidence level): '))
            tTest(slope, slope_std_error, degreesFreedom, a)                
        elif menuInput == 6:# fTest
            if SSR==-1:
                SSR=float(input('Enter SSR: '))
            if SSE ==-1:
                SSE=float(input('Enter SSE: '))
            if not data:
                degreesFreedom = float(input('Enter degreesFreedom: '))
                a = float(input('Enter alpha (confidence level): '))
            fTest(SSR, SSE, degreesFreedom, a)
        elif menuInput == 7:# autocorrelation
            if data:
                e = getErrors(intercept, slope,  x, y)
                autoCorr(x,e)
            else:
                e = np.array([])
                eInput = input('\nInput e values : ')
                eInput=eInput.split()
                for i in eInput:
                    e = np.append(e, float(i))
                time = np.array([])
                timeInput = input('\nInput time values : ')
                timeInput=timeInput.split()
                for i in timeInput:
                    time = np.append(time, float(i))
                autoCorr(time,e)
        elif menuInput == 8:#residual std error
            if data:
                predError = y - predictY(intercept, slope,  x)
                residualSE = np.sqrt(np.sum(predError**2)/degreesFreedom)
                print('residual std error = %6.6f'%residualSE)
            else:
                print('Stats need to be provided for this option.')          
        elif menuInput == 9:#cooef of correlation
            if data:
                corrCoef=np.corrcoef(x,y)
                corrCoef=corrCoef[0][1]
                print('the coeficient of correlation is ' + str(corrCoef) +
                      '. \nThis indicates ')
                if corrCoef > a:
                    if corrCoef > .75:
                        print('very ')
                    print('a positive ')
                elif corrCoef < (-1*a):
                    if corrCoef < -.75:
                        print('very ')
                    print('a negative ')
                elif corrCoef < (-1*a):
                    print('no or practically no ')
                print('correlation.\n')
                        
            else:
                print('Stats need to be provided for this option.')  
        elif menuInput == 10:#population mean response intervals
            goOn=False
            if not data:
                n=int(input('Enter n: '))
                intercept=float(input('Enter intercept B0: '))
                slope=float(input('Enter slope B1: '))
                SYX=float(input('Enter Standard Error SYX: '))
                xBar=float(input('Enter x mean, xBar: '))
                SSX=float(input('Enter SSX, sum of square x df from mean x (or -1 to enter h instead): '))
                if SSX==-1:
                    h=float(input('Enter h stat: '))
                a = float(input('Enter alpha: '))
                goOn=True
            else:
                if SYX!=-1:
                    n=x.size
                    xBar=np.mean(x)
                    SSX = 0.0
                    for i in x:
                        SSX+=((i-xBar)**2)
                    goOn=True
                else:
                    print('Get Sum of Squares first')
            if goOn:
                testX=float(input('Enter x to test: '))
                confidence=(1-a)*100
                prediction=predictY(intercept, slope, testX)
                if SSX != -1:
                    h=(1/n)+((testX-xBar)**2/SSX)
                tCrit = stats.t.ppf(1-(a/2), n-2)
                yLower=prediction-(tCrit*SYX*(h**(1/2)))
                yUpper=prediction+(tCrit*SYX*(h**(1/2)))
                print('With %2.2f percent confidence, the mean responce for when X=%2.2f is:\n%6.6f<= mean Y <= %6.6f' %(confidence, testX, yLower, yUpper))
                predictLower=prediction-(tCrit*SYX*((1+h)**(1/2)))
                predictUpper=prediction+(tCrit*SYX*((1+h)**(1/2)))
                print('The prediction interval is:\n%6.6f <=  X=%2.2f ->Y <= %6.6f'%(predictLower, testX, predictUpper))
        elif menuInput == 11:#rSquared
            if not data:
                SSR=float(input('Enter SSR: '))
                SSE=float(input('Enter SSE: '))
                rSquared=(SSR/(SSR+SSE))
            elif SSR == -1:
                SSR, SSE, SST, rSquared, SYX = sumSquares(x,y, intercept, slope)
            print('rSquared = %6.6f.  This is means %6.6f percent of the variation in the dependent variable can be explained by variation in the independent variable.' % (rSquared,(100*rSquared)))
        elif menuInput == 12:#quit
            break
    




##############END MAIN###############################
