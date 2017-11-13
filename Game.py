import numpy as np
from enum import Enum
import cPickle as pickle

w = 10
h = 10

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
def move_coord(dir, x, y, max_x, max_y):
    if (dir == Direction.UP):
        if y == max_y:
            return (x, y)
        else:
            return (x, y+1)
    if (dir == Direction.DOWN):
        if y == 0:
            return (x, y)
        else:
            return (x, y-1)
    if (dir == Direction.LEFT):
        if x == 0:
            return (x, y)
        else:
            return (x-1, y)
    if (dir == Direction.RIGHT):
        if x == max_x:
            return (x, y)
        else:
            return (x+1, y)

playfield = np.full((w, h), 0)
ind1 = np.random.randint(0, w-1)
ind2 = np.random.randint(0, h-1)

playfield[ind1, ind2] = 1
        
x = ind1
y = ind2

simulations = {}
            
for j in range(1000):
    for i in range(11):
        playfield = np.full((w,h), 0)
        playfield[x, y] = 1
        dir = np.random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])
        simulations[(j, i)] = (playfield, dir)
        (x, y) = move_coord(dir, x, y, w-1, h-1)
        
sim_file = open("simulations", "w")
pickle.dump(simulations, sim_file)
sim_file.close()