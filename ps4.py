import numpy as np
import sys

###Problem 1
###Provided function to create training data
def simplest_training_data(n):
  w = 3
  b = 2
  x = np.random.uniform(0,1,n)
  y = 3*x+b+0.3*np.random.normal(0,1,n)
  return (x,y)

def simplest_training(n, k, eta):
   #TODO: Your Code Here

  w = np.random.normal(0,1)

  b = 0

  training_data = simplest_training_data(n)

  #print(training_data[0])
  #print(training_data[1])

  for x in range(1, k):

    wd = 0

    bd = 0

    for i in range(0,n-1):

        wd = wd + (-2*training_data[0][i]) * (training_data[1][i] - (w*training_data[0][i] + b))

        bd = bd + (-2*(training_data[1][i] - (w*training_data[0][i] + b)))

    w = w - (wd/n)*eta

    b = b - (bd/n)*eta
 
  theta = (w,b)

  #print(theta)
  return theta




def simplest_testing(theta, x):
  # To test, I called simplest_training to get the output of the training algorithim, and then looped from 0 to x and ensured that the 
  # resulting y coordinate gotten from multiplying the value by the weight and adding the bias followed the line 3x + 2
  y = []
  data = simplest_training(theta[0], theta[1], theta[2])

  for i in range(0,x):

    y.append(i*data[0] + data[1]) 

  return y

#run it all
#theta = simplest_training(10,10000,.01)
print(simplest_testing([30,10000,.02], 10))

###Problem 2
###Provided function to create training data
def single_layer_training_data(trainset):
  n = 10
  if trainset == 1:
    # Linearly separable
    X = np.concatenate((np.random.normal((0,0),1,(n,2)), np.random.normal((10,10),1,(n,2))),axis=0)
    y = np.concatenate((np.ones(n), np.zeros(n)),axis=0)

  elif trainset == 2:
    # Not Linearly Separable
    X = np.concatenate((np.random.normal((0,0),1,(n,2)), np.random.normal((10,10),1,(n,2)), np.random.normal((10,0),1,(n,2)), np.random.normal((0,10),1,(n,2))),axis=0)
    y = np.concatenate((np.ones(2*n), np.zeros(2*n)), axis=0)

  else:
    print("function single_layer_training_data undefined for input", trainset)
    sys.exit()

  return (X,y)

def sigmoid(x, w):

  z = np.dot(x, w)

  return 1/(1 + np.exp(-z))
    

def single_layer_training(k, eta, trainset):
  '''
  TRAINSET = 1 confidently classifies the data. This is because it is linearlly seperable. Values near (0,0) are assigned a probability of 
  or very close to 1 for class 1, and values near (10,10) got probabilities of or close to 0. 
  TRAINSET = 2 is much less confident when classifying: while some probabilities are close to 0 or 1, some are in the middle, demonstrating 
  uncertainty. This is due to the fact that the data is not linearlly seperable.

  '''
  data = single_layer_training_data(trainset)

  if trainset == 1:

    n = 20

  else:

    n = 40
  
  w1 = np.random.normal(0,1)

  w2 = np.random.normal(0,1)

  b = 0
    
  for x in range(k):
    
    wd1 = 0

    wd2 = 0

    bd = 0

    for i in range(0, n - 1):

      x1 = data[0][i][0]

      x2 = data[0][i][1]

      y = data[1][i]

      z = (w1 * x1) + (w2 * x2) + b

      if data[1][i] == 1:

        wd1 = wd1 - (-x1 * (np.exp(-z))) / (1 + (np.exp(-z)))

        wd2 = wd2 - (-x2 * (np.exp(-z))) / (1 + (np.exp(-z)))

        bd = bd - (-(np.exp(-(z)))) / (1 + np.exp(-z))

      elif data[1][i] == 0:

        wd1 = wd1 - ((-x1) / (np.exp(-z) + 1))

        wd2 = wd2 - ((-x2) / (np.exp(-z) + 1))

        bd = bd - ((-1) / (np.exp(-z) + 1))

    w1 = w1 - (wd1/n) * eta

    w2 = w2 - (wd2/n) * eta

    b = b - (bd/n) * eta

      
 
  theta = (w1, w2, b)

  #print("Theta: ")

  #print(theta)

  return theta

def single_layer_testing(theta, X):
  #TODO: Your Code Here
  ans = []

  w1 = theta[0]

  w2 = theta[1]

  b = theta[2]

  for i in range(0,len(X)-1):

    za = -(w1*X[i][0]) + (w2*X[i][1]) + b

    ans.append(1/(1 + np.exp(za)))

  #print("PROBABILITIES:")

  #print(ans)

  y = ans

  return y

#now run
#single_layer_training(10000, .000001, 2)

print(single_layer_testing(single_layer_training(100,.01,1),single_layer_training_data(1)[0]))
print()
print(single_layer_testing(single_layer_training(10000,.00001,2),single_layer_training_data(2)[0]))

###Problem 3
###Provided function to create training data
def pca_training_data(n, sigma):
  m = 1
  b = 1
  x1 = np.random.uniform(0,10,n)
  x2 = m*x1+b
  X = np.array([x1,x2]).T
  X += np.random.normal(0,sigma,X.shape)
  return X

def pca_training(k, eta, n, sigma):

  w11 = np.random.normal(0,1)

  w12 = np.random.normal(0,1)

  w21 = np.random.normal(0,1)

  w22 = np.random.normal(0,1)

  b11 = 0

  b21 = 0

  b22 = 0

  training_data = pca_training_data(n, sigma)

  for x in range(k):

    wd11 = 0

    wd12 = 0

    wd21 = 0

    wd22 = 0

    bd11 = 0

    bd21 = 0

    bd22 = 0

    for i in range(0,n-1):

      x1 = training_data[i][0]

      x2 = training_data[i][1]


      h = (w11 * x1) + (w12 * x2) + b11

      wd11 = wd11 + (2 * ( (w21 * h)  + b21 - x1) ) * (w21 * x1) + (2 * (w22 * ( h ) + b22 - x2) * x1 * w22 )

      wd12 = wd12 + (2 * ((w21 * h) + b21 - x1 )) * (w21 * x2) + (2 * (w22 * (w22 * (h) ) +b22 - x2 ) ) * x2 * w22

      bd11 =  bd11 + (2 * ((w21 * h ) + b21 - x1)) * w21 + (2 * (w22 * h) + b22 - x2) * w22

      

      wd21 = wd21 + (2 * (w21 * h + b21 -x1) ) * h 

      wd22 = wd22 + (2 * (w22 * h + b22 -x2) ) * h

      bd21 = bd21 + (2 * (w21 * h + b22 -x1) )

      bd22 = bd22 + (2 * (w22 * h + b22 -x2) )



    w11 = w11 - (wd11 / n) * eta

    w12 = w12 - (wd12 / n) * eta

    b11 = b11 - (bd11/n) * eta



    w21 = w21 - (wd21/n) * eta

    w22 = w22 - (wd22/n) * eta

    b21 = b21 - (bd21/n) * eta

    b22 = b22 - (bd22/n) * eta


  theta = ([w11, w12, w21, w22], [b11, b21, b22])

  #print(theta)
  return theta

def pca_test(theta, X):
  #TODO: Your Code Here
  y = []

  for i in range(len(X)):

    z1 = theta[0][2]*(theta[0][0]*X[i][0] + theta[0][1]*X[i][1] + theta[1][0]) + theta[1][1]

    z2 = theta[0][3]*(theta[0][0]*X[i][0] + theta[0][1]*X[i][1] + theta[1][0]) + theta[1][2]

    y.append((z1,z2))

  #print(X)
  #print(y)

  return y


#pca_training(10000, .00000001, 100, 2)
print(pca_test(pca_training(10000, .0001, 10, .1), [[1,2],[4,5],[10,3]]))


###Problem 4: Challenge Problem
def nn_training(k, eta, trainset, H):
  #TODO: Your Code Here
  return theta

def nn_testing(theta, X):
  #TODO: Your Code Here
  return y