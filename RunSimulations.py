from Game import *
from Backpropagation import *
import pickle
import numpy as np

print("Imports completed. Starting...")

sim_file = open("simulations", "r")
simulations = pickle.load(sim_file)

print("Simulations loaded.")

def direction_to_vector(dir):
    if dir == Direction.UP:
        return np.array([[1], [0], [0], [0]])
    if dir == Direction.DOWN:
        return np.array([[0], [1], [0], [0]])
    if dir == Direction.LEFT:
        return np.array([[0], [0], [1], [0]])
    if dir == Direction.RIGHT:
        return np.array([[0], [0], [0], [1]])
    return None
        

print("Generating input and output matrices.")

iopairs = []
for i in range(1000):
    sim_input_vector = None
    for j in range(10):
        simulation_input = simulations[(i, j)]
        sim_input_vector1 = simulation_input[0].flatten().reshape(100, 1)
        sim_input_vector2 = direction_to_vector(simulation_input[1])
        if sim_input_vector != None:
            sim_input_vector = np.concatenate((sim_input_vector, sim_input_vector1, sim_input_vector2), axis=0)
        else:
            sim_input_vector = np.concatenate((sim_input_vector1, sim_input_vector2), axis=0)
    #sim_input_vector = np.concatenate((sim_input_vector, simulations[i, 10][0].flatten().reshape(100,1)), axis=0)
    sim_input_vector = np.concatenate((sim_input_vector, np.array([[1]])))
    sim_output_vector = simulations[(i, 10)][0].flatten()
    iopairs.append((sim_input_vector, sim_output_vector))
    
print(iopairs[0][0].shape)
print(iopairs[0][1].shape)

print("Input and output matrices generated.")
print("Forming neural network with one hidden layer and random connections.")

randomnet = RandomNet(iopairs[0][0].shape[0]-1, int((iopairs[0][0].shape[0]-1)/2), iopairs[0][1].shape[0])

print("Neural network ready. Begin training.")

readynet = backpropagate(0.3, 0.3, [input for (input,output) in iopairs], [output for (input, output) in iopairs], randomnet, 3)

print("Network has been trained. Testing...")

i = 1
sim_input_vector = None
for j in range(10):
    simulation_input = simulations[(i, j)]
    sim_input_vector1 = simulation_input[0].flatten().reshape(100, 1)
    sim_input_vector2 = direction_to_vector(simulation_input[1])
    if sim_input_vector != None:
        sim_input_vector = np.concatenate((sim_input_vector, sim_input_vector1, sim_input_vector2), axis=0)
    else:
        sim_input_vector = np.concatenate((sim_input_vector1, sim_input_vector2), axis=0)
#sim_input_vector = np.concatenate((sim_input_vector, simulations[i, 10][0].flatten().reshape(100,1)), axis=0)
sim_input_vector = np.concatenate((sim_input_vector, np.array([[1]])))

print("Output:")
print(readynet.calculate_output(sim_input_vector))