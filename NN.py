import numpy as np
import unittest

class SimpleNet(object):
    
    def __init__(self, function, *weight_matrices):
        self.function = function
        self.weight_matrices = weight_matrices
        
    def process(self, vector):
        vector_updated = np.append(vector, 1)
        for weight_matrix in self.weight_matrices:
            vector_updated = np.dot(weight_matrix,vector_updated)
            vector_updated = np.append(vector_updated, 1)
        return vector_updated
        
class BackpropagatedNet(SimpleNet):
    def backpropagate(input_list, output_list):
        print("Not implemented yet.")
        # To be implemented
        
class TestSimpleNet(unittest.TestCase):
    
    def test_xor_net(self):
        #xornet = SimpleNet(np.array([[-1, -1, 1],[-1, -1, 1]]), np.array([x, y, 1]))
        xornet = SimpleNet(np.array([[-1, -1, 1],[-1, -1, 1]]))
        self.assertEqual(xornet.process(np.array([0, 0])), 0)
        self.assertEqual(xornet.process(np.array([1, 0])), 1)
        self.assertEqual(xornet.process(np.array([0, 1])), 1)
        self.assertEqual(xornet.process(np.array([1, 1])), 0)
        
if __name__ == '__main__':
    unittest.main()