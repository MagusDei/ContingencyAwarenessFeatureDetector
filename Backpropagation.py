import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg,stats
import time

def numplot():
    mat0 = np.matrix([[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],[0,1,0,0,0,1,0],[0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]])
    mat1 = np.matrix([[0,0,0,0,0,0,0],[0,0,0,1,0,0,0],[0,0,1,1,0,0,0],
    [0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]])
    mat2 = np.matrix([[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0],[0,1,1,1,1,1,0],[0,1,0,0,0,0,0],
    [0,1,0,0,0,0,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]])
    mat3 = np.matrix([[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0],[0,0,0,1,1,1,0],[0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]])
    mat4 = np.matrix([[0,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,1,0,0,0,0,0],
    [0,1,0,1,0,0,0],[0,1,1,1,1,0,0],[0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,0,0]])
    mat5 = np.matrix([[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,1,0,0,0,0,0],
    [0,1,0,0,0,0,0],[0,1,1,1,1,1,0],[0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]])
    mat6 = np.matrix([[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,1,0,0,0,0,0],
    [0,1,0,0,0,0,0],[0,1,1,1,1,1,0],[0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]])
    mat7 = np.matrix([[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,0,0,0,0,1,0],
    [0,0,0,0,1,0,0],[0,0,0,1,0,0,0],[0,0,1,0,0,0,0],
    [0,1,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,0,0,0,0,0]])
    mat8 = np.matrix([[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],[0,1,1,1,1,1,0],[0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]])
    mat9 = np.matrix([[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,0]])
    plt.imshow(mat9, interpolation='nearest')
    plt.show()

def simpleplot():
    X = [0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 0.95, 1.10]
    Y = [1.60, 1.40, 1.40, 1.60, 1.70, 2.00, 1.70, 2.10]
    plt.plot(X, Y, c="blue", linestyle=" ", marker=".", markersize = 5)
    plt.show()

def sampleWidrowHoffLearner():
    weightmatrix = np.matrix([[0.5, 0.5]])
    perceptron = SimplePerceptron(lambda x: x,weightmatrix)
    X = [0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 0.95, 1.10]
    Y = [1.60, 1.40, 1.40, 1.60, 1.70, 2.00, 1.70, 2.10]
    for i,input in enumerate(X):
       y2 = perceptron.calculate_output(np.matrix([[input],[1]]))
       err = Y[i]-y2
       if err == 0:
           break
       currwm = perceptron.weightmatrices[0]
       for x in np.nditer(currwm, op_flags=['readwrite']):
           x[...] = x+0.3*err*x
       perceptron.weightmatrices[0] = currwm
    Y2 = [perceptron.calculate_output(np.matrix([[i],[1]])).item(0) \
             for i in X]
    plt.plot(X, Y, c="blue", linestyle=" ", marker=".", markersize = 5)
    plt.plot(X, Y2, c="red", linestyle="-")
    plt.show()

def backpropagate(rate, tolerance, inputs, targets, net, maxrep=10000):
    """ Implements a backpropagation loop
    
    >>> inputs = [np.matrix('1.0;1.0;1.0'), np.matrix('1.0;0.0;1.0'),
    ...                       np.matrix('0.0;1.0;1.0'), np.matrix('0.0;0.0;1.0')]
    >>> ortargets = [np.matrix('1.0'), np.matrix('1.0'),
    ...                            np.matrix('1.0'), np.matrix('0.0')]
    >>> andtargets = [np.matrix('1.0'), np.matrix('0.0'),
    ...                               np.matrix('0.0'), np.matrix('0.0')]
    >>> orperceptron = SimplePerceptron(lambda x: (1/(1+np.exp(-x))),
    ...       np.matrix('6.0,6.0,-3.0'))
    >>> all([np.allclose(orperceptron.calculate_output(inputs[i]),
    ...       ortargets[i], atol=0.1) for i in range(len(inputs))])
    True

    >>> orperceptron = backpropagate(0.4, 0.1, inputs, andtargets, orperceptron)
    >>> all([np.allclose((orperceptron.calculate_all_outputs(inputs[i]))[-1],
    ...       andtargets[i], atol=0.1) for i in range(len(inputs))])
    True
    """
    assert len(inputs) == len(targets)
    converge = False
    inum = 0
    while not converge and inum < maxrep:
        inum += 1
        converge = True
        for input,target in zip(inputs, targets):
            #Calculate outputs, check against target, backpropagate
            outputs = net.calculate_all_outputs(input)
            if np.allclose(outputs[-1], target, atol=tolerance):
                continue # Continue for loop without backpropagating
            else:
                converge = False
                net = backpropagate_single(rate, outputs, target, net)
            #print("Hello World!")
    return net

def backpropagate_single(rate, outputs, target, net):
    """Implements a single forward-backward pass
    of backpropagation for later looping.

    >>> neuralnet = SimplePerceptron(lambda x: (1/(1+np.exp(-x))), 
    ...      np.matrix([[-2, -2, 2],[3,3,2]]), 
    ...      np.matrix([[-2,-4,3],[2,2,-2]]),
    ...      np.matrix([[3,1,-2]]))
    >>> input = np.matrix('0.1;0.9;1')
    >>> outputs = neuralnet.calculate_all_outputs(input)
    >>> round(outputs[-1].item(0), 3)
    0.288
    >>> target = np.matrix('0.9')
    >>> neuralnet = backpropagate_single(0.8, outputs, target, neuralnet)
    >>> curr = neuralnet.weightmatrices[0]
    >>> w = np.matrix([[-2.001, -2.005,  1.994],
    ...        [ 2.999,  2.999,  1.999]])
    >>> np.allclose(curr, w, atol=0.001)
    True
    >>> curr = neuralnet.weightmatrices[1]
    >>> w = np.matrix([[-1.984, -3.968,  3.032],
    ...        [ 2.010,  2.019,  -1.980]])
    >>> np.allclose(curr, w, atol=0.001)
    True
    >>> curr = neuralnet.weightmatrices[2]
    >>> w = np.matrix([[3.012, 1.073,  -1.900]])
    >>> np.allclose(curr, w, atol=0.001)
    True
    """

    output = outputs[-1]
    #output = np.delete(outputs[-1], len(outputs[-1])-1)
    outputerr = np.multiply((target-output), output)
    outputerr = np.multiply(outputerr, (np.ones_like(output)-output))
    #outputerr = outputerr.transpose()
    for i,wmat in reversed(list(enumerate(net.weightmatrices))):
        newmat = np.ones_like(wmat)
        newmat = np.multiply(newmat, outputerr.transpose())
        #print(newmat)
        newmat = np.multiply(newmat, outputs[i].transpose())
        #print(newmat)
        newmat = np.multiply(rate, newmat)
        net.weightmatrices[i] = net.weightmatrices[i]+newmat
        #print(outputerr)
        outputerr = outputerr*wmat
        oj = outputs[i]
        factor = np.multiply(oj, (np.ones_like(oj)-oj))
        outputerr = np.multiply(outputerr, factor.transpose())
        outputerr = np.delete(outputerr, -1)
    return net

def RandomNet(inputs, *nofneurons):
    """Creates a neural network with the desired layer
    architecture and initializes it with random weights
    in the range [-0.3,0.3]. The activation function is
    sigmoid.

    >>> randomnetwork = RandomNet(2, 2, 1)
    >>> len(randomnetwork.weightmatrices)
    2

    >>> int(round(-np.log((1/randomnetwork.activ(4))-1)))
    4

    >>> type(randomnetwork.weightmatrices[0]) is np.matrix
    True

    >>> randomnetwork.weightmatrices[0].shape
    (2, 3)

    >>> randomnetwork.weightmatrices[1].shape
    (1, 3)

    """
    randomnet = SimplePerceptron(lambda x: (1/(1+np.exp(-x))))
    lastn = inputs+1
    for n in nofneurons:
        weightmatrix = np.asmatrix((6*np.random.rand(n,lastn))-3)
        randomnet.weightmatrices.append(weightmatrix)
        lastn = n+1
    return randomnet

class SimplePerceptron():
    """A class for implementing a network of perceptrons.
    
    >>> oneperceptron = SimplePerceptron(lambda x: (x>0), np.matrix('2'))
    >>> mat = np.matrix('1')
    >>> s = oneperceptron.calculate_output(mat)
    >>> print(s)
    [[1]]

    >>> mat2 = np.matrix('-1')
    >>> s2 = oneperceptron.calculate_output(mat2)
    >>> print(s2)
    [[0]]
    """

    def __init__(self, activation_function, *weightmatrices):
        self.activ = activation_function
        self.weightmatrices = list(weightmatrices)

    def calculate_output(self, input):
        for wmat in self.weightmatrices:
            input = wmat*input
            for i in range(len(input)):
                input[i] = self.activ(input.item(i))
                #if not self.activ(input.item(i)):
                    #input[i] = 0
                #else:
                    #input[i] = 1
            input = np.concatenate((input, [[1]]))
        input = np.delete(input, len(input)-1)
        return input

    def calculate_all_outputs(self, input):
    # Do a forward pass, calculating and storing outputs at each layer
        outputs = [input]
        for wmat in self.weightmatrices:
            input = wmat*input
            for i in range(len(input)):
                input[i] = self.activ(input.item(i))
            input = np.concatenate((input, [[1]]))
            outputs.append(input)
        outputs[-1] = np.delete(outputs[-1], len(outputs[-1])-1)
        return outputs

def xor(x1, x2):
    """Implements XOR as a perceptron architecture

    >>> xor(1, 1)
    0
    >>> xor(1, 0)
    1
    >>> xor(0, 1)
    1
    >>> xor(0, 0)
    0
    """
    weightmatrices = [np.matrix('-1, -1, 1.5; -1, -1, 0.5'), np.matrix('1, -1, -0.5')]
    #return weightmatrices
    xorperceptron = SimplePerceptron(lambda x: int(x>0), weightmatrices[0], weightmatrices[1])
    #return xorperceptron.weightmatrices

    input = np.matrix([[x1], [x2], [1]])
    #return weightmatrices[0]*input
    output = xorperceptron.calculate_output(input)

    return int(output.item(0))

if __name__ == '__main__':
    import doctest
    doctest.testmod()