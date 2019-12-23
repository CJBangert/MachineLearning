import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

SIGMA = .1

def getOnesCol():

    return np.ones(21)

def getY():

    noise = 1

    x = np.arange(0, 1.05, .05)

    y = []

    for i in range(len(x)):

        if abs(x[i] - 0.5) < 0.25:

            y.append((x[i] + 1) + np.random.normal(0, noise))

        else:

            y.append(-x[i]  + np.random.normal(0, noise)) 

    return y


def leastSquaresLineFit(data):

    oneCol = getOnesCol()

    Y = np.transpose(data[1])

    X = np.array(np.transpose([np.array(oneCol), np.array(data[0])]))
    
    XtX= np.transpose(X) @ X

    XtY = np.transpose(X) @ Y

    XtXI = np.linalg.inv(XtX)
    
    regressionLine = (XtXI) @ (XtY)

    return regressionLine


def testLeastSquaresLineFit():

    testOne = [np.arange(0, 1.05, 0.05), np.arange(1, 3.1, 0.1)]

    var = np.arange(0, 1.05, 0.05)

    y2 = np.dot(var, 2) + 1 + np.dot(np.random.randn(len(var)), .01)

    testTwo = [var, y2]

    y3 = getY()

    testThree = [var, y3]

    firstLine = leastSquaresLineFit(testOne)

    secondLine = leastSquaresLineFit(testTwo)

    three = leastSquaresLineFit(testThree)

    line1 = np.dot(firstLine[1], var) + firstLine[0]

    line2 = np.dot(secondLine[1], var) + secondLine[0]

    line3 = np.dot(three[1], var) + three[0]

    plt.plot(testOne[0], testOne[1], "o")

    plt.plot(testOne[0], line1)

    plt.savefig("plot1.png")

    plt.close()

    plt.plot(var, y2, 'o')

    plt.plot(var,line2)

    plt.savefig('plot2.png')

    plt.close()

    plt.plot(var,y3, 'o')

    plt.plot(var,line3)

    plt.savefig('plot3.png')



def weightedLeastSquaresFit(x, wx, wy):
    
    regressionLine = []
    
    onesCol = getOnesCol()
    
    WY = wy
    
    WX = wx
    
    X = np.transpose([np.array(onesCol), np.array(x)])
      
    XtX= np.transpose(X) @ WX
    
    XtY = np.transpose(X) @ WY
    
    XtXI = np.linalg.inv(XtX)
    
    regressionLine = XtXI @ XtY
    
    return regressionLine

def plot_iter(name, a1, b1, a2, b2, x, y, w1, w2, name1, name2):

    slope1 = a1

    int1 = b1

    slope2 = a2

    int2 = b2

    ab1 = np.dot(x, slope1) + int1

    ab2 = np.dot(x, slope2) + int2

    plt.scatter(x,y)

    plt.plot(x, ab1)

    plt.plot(x, ab2)

    plt.savefig(name)

    plt.close()

    plt.scatter(x, w1)

    itername = name1

    plt.savefig(itername)

    plt.close()

    plt.scatter(x, w2)

    itername =  name2

    plt.savefig(itername)

    plt.close()

def expectation(x, y, a1, b1, a2, b2, count):

    w = []

    for i in range(len(y)):
        #goal is to minimize residuals
        
        y1 = a1 * x[i] + b1

        y2 = a2 * x[i] + b2

        r1 = y1 - y[i]

        r2 = y2 - y[i]

        r1 = r1 * r1

        r2 = r2 * r2

        newB = np.exp(-r2 / SIGMA) 

        newA = np.exp(-r1 / SIGMA)

        w1 = newA / (newA + newB)

        w2 = newB / (newA + newB)

        w.append((w1,w2))


    maximize(x, y, w, a1, b1, a2, b2, count)

def maximize(x, y, w, a1, b1, a2, b2, count):

    tempa1 = a1

    tempb1 = b1

    tempa2 = a2

    tempb2 = b2

    w1 = []

    w2 = []

    wx1 = []

    wy1 =[]

    wx2 = []

    wy2 = []

    onesColumn = getOnesCol()

    for i in range(len(y)):

        w1.append(w[i][0])

        w2.append(w[i][1])

    for i in range(len(y)):

        wy1.append(w1[i] * y[i])
                    
    for i in range(len(y)):

        wy2.append(w2[i] * y[i])

    wx1 = np.transpose([np.array(onesColumn * w1), np.array(x * w1)])

    wx2 = np.transpose([np.array(onesColumn * w2), np.array(x * w2)])

    firstLine = weightedLeastSquaresFit(x, wx1, wy1)

    secondLine = weightedLeastSquaresFit(x, wx2, wy2)

    a1 = firstLine[1]

    b1 = firstLine[0]

    a2 = secondLine[1]

    b2 = secondLine[0]

    if (tempa1 == a1 and tempb1 == b1  and tempb2 == b2 and tempa2 == a2):

        #converged

        print((a1,b1), (a2,b2))

        ab1 = np.dot(x, a1) + b1

        ab2 = np.dot(x, a2) + b2

        plt.scatter(x,y3)

        plt.plot(x, ab1)

        plt.plot(x, ab2)

        plt.savefig('finalPlot.png')

        plt.close()



    else:

        if count == 1:

            plot_iter('iter1.png',a1, b1, a2, b2, x, y3, w1, w2, 'iter1weight1.png','iter1weight2.png')

            
        elif count == 2:

            plot_iter('iter2.png',a1, b1, a2, b2, x, y3, w1, w2,'iter2weight1.png','iter2weight2.png')

        elif count == 3:

            plot_iter('iter3.png',a1, b1, a2, b2, x, y3, w1, w2, 'iter2weight3.png','iter3weight3.png')
            
        elif count == 4:

           plot_iter('iter4.png',a1, b1, a2, b2, x, y3, w1, w2, 'iter4weight1.png','iter4weight2.png')

        elif count == 5:

            plot_iter('iter5.png',a1, b1, a2, b2, x, y3, w1, w2, 'iter5weight1.png','iter5weight2.png')

        count = count + 1

        expectation(x, y, a1, b1, a2, b2, count)


x3 = np.arange(0, 1.05, .05)

y3 = getY()


a1 = np.random.normal(0, 1)

b1 = np.random.normal(0, 1)

a2 = np.random.normal(0, 1)

b2 = np.random.normal(0, 1)

count = 1


testLeastSquaresLineFit()

expectation(x3, y3, a1, b1, a2, b2, count)
