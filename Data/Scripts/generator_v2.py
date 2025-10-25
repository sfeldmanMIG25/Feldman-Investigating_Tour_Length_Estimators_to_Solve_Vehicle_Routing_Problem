# generator.py
"""
VRP Instance Generator (Proportional Scaling)

This script generates VRP instances with parameters that scale relative to the
total number of customers (n), making it suitable for creating very large,
customizable instances.

The key changes are:
- The capacity is no longer a fixed value but a proportion of the total demand.
- The number of seeds for clustered customer generation scales with n.
- The `avgRouteSize` parameter from the original script has been re-conceptualized
  as a capacity factor, allowing direct control over the average route load.
"""
import sys
import random
import math
import os
import numpy as np

def distance(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

# Global constants
maxCoord = 1000
decay = 40

# Redesigned input argument handling
if len(sys.argv) < 8:
    print('Missing arguments:\n\t python generator.py n depotPos custPos demandType capacityFactor instanceID randSeed')
    help = """
    n (number of customers, e.g., 100, 1000, 10000)

    Depot positioning
        1 = Random
        2 = Centered
        3 = Cornered

    Customer positioning
        1 = Random
        2 = Clustered
        3 = Random-clustered

    Demand distribution
        1 = Unitary
        2 = Small, large var
        3 = Small, small var
        4 = Large, large var
        5 = Large, small var
        6 = Large, depending on quadrant
        7 = Few large, many small

    Capacity factor (e.g., 0.1 for 10% of total demand, 0.05 for 5%)
        This replaces the old 'avgRouteSize' and gives precise control.

    instanceID
    randSeed (random seed for reproducibility)
    """
    print(help)
    exit(0)

# Read input arguments
n = int(sys.argv[1])
rootPos = int(sys.argv[2])
custPos = int(sys.argv[3])
demandType = int(sys.argv[4])
capacityFactor = float(sys.argv[5])
instanceID = int(sys.argv[6])
randSeed = int(sys.argv[7])

random.seed(randSeed)

# Depot positioning
depot = (-1, -1)
if rootPos == 1:
    depot = (random.randint(0, maxCoord), random.randint(0, maxCoord))
elif rootPos == 2:
    depot = (int(maxCoord / 2.0), int(maxCoord / 2.0))
elif rootPos == 3:
    depot = (0, 0)
else:
    print("Depot Positioning out of range!")
    exit(0)

# Customer positioning
S = set()
nRandCust = 0
nSeeds = 0

if custPos == 1:
    nRandCust = n
elif custPos == 2:
    nSeeds = max(2, int(0.01 * n)) # Scale seeds with n (e.g., 1% of n)
elif custPos == 3:
    nRandCust = int(n / 2.0)
    nSeeds = max(2, int(0.01 * n))
else:
    print("Customer Positioning out of range!")
    exit(0)

nClustCust = n - nRandCust

# Generating random customers
for _ in range(nRandCust):
    x, y = random.randint(0, maxCoord), random.randint(0, maxCoord)
    while (x, y) in S or (x, y) == depot:
        x, y = random.randint(0, maxCoord), random.randint(0, maxCoord)
    S.add((x, y))

# Generation of the clustered customers
seeds = []
if nClustCust > 0:
    # Generate the seeds
    for _ in range(nSeeds):
        x, y = random.randint(0, maxCoord), random.randint(0, maxCoord)
        while (x, y) in S or (x, y) == depot:
            x, y = random.randint(0, maxCoord), random.randint(0, maxCoord)
        S.add((x, y))
        seeds.append((x, y))

    # Determine the seed with maximum sum of weights
    maxWeight = 0.0
    for i_seed, j_seed in seeds:
        w_ij = sum(2**(-distance((i_seed, j_seed), (i_, j_)) / decay) for i_, j_ in seeds)
        if w_ij > maxWeight:
            maxWeight = w_ij
    norm_factor = 1.0 / maxWeight

    # Generate the remaining customers using Accept-reject method
    while len(S) < n:
        x, y = random.randint(0, maxCoord), random.randint(0, maxCoord)
        if (x, y) in S or (x, y) == depot:
            continue
        
        weight = sum(2**(-distance((x, y), (i_, j_)) / decay) for i_, j_ in seeds) * norm_factor
        
        if random.uniform(0, 1) <= weight:
            S.add((x, y))

V = [depot] + list(S)
D = []
sumDemands = 0
maxDemand = 0

# Demands
demandMinValues = [1, 1, 5, 1, 50, 1, 51, 50, 1]
demandMaxValues = [1, 10, 10, 100, 100, 50, 100, 10]
demandMin = demandMinValues[demandType - 1]
demandMax = demandMaxValues[demandType - 1]
demandMinEvenQuadrant = 51
demandMaxEvenQuadrant = 100
demandMinLarge = 50
demandMaxLarge = 100
largePerRoute = 1.5
demandMinSmall = 1
demandMaxSmall = 10

for i in range(n):
    j = int((demandMax - demandMin + 1) * random.uniform(0, 1) + demandMin)
    if demandType == 6:
        v_coords = V[i + 1]
        if (v_coords[0] < maxCoord / 2.0 and v_coords[1] < maxCoord / 2.0) or (v_coords[0] >= maxCoord / 2.0 and v_coords[1] >= maxCoord / 2.0):
            j = int((demandMaxEvenQuadrant - demandMinEvenQuadrant + 1) * random.uniform(0, 1) + demandMinEvenQuadrant)
    if demandType == 7:
        if i < (n * largePerRoute / (1/capacityFactor)):
            j = int((demandMaxLarge - demandMinLarge + 1) * random.uniform(0, 1) + demandMinLarge)
        else:
            j = int((demandMaxSmall - demandMinSmall + 1) * random.uniform(0, 1) + demandMinSmall)
    D.append(j)
    if j > maxDemand:
        maxDemand = j
    sumDemands += j

# Generate capacity based on total demand and capacity factor
capacity = math.ceil(sumDemands * capacityFactor)
# Ensure capacity is at least enough for the largest single demand
if maxDemand > capacity:
    capacity = maxDemand

instanceName = f"XML{n}_{rootPos}{custPos}{demandType}_{capacityFactor}_{instanceID:02d}"

pathToWrite = instanceName + '.vrp'

with open(pathToWrite, 'w') as f:
    f.write('NAME : ' + instanceName + '\n')
    f.write('COMMENT : Generated with proportional scaling\n')
    f.write('TYPE : CVRP\n')
    f.write('DIMENSION : ' + str(n + 1) + '\n')
    f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
    f.write('CAPACITY : ' + str(int(capacity)) + '\n')
    f.write('NODE_COORD_SECTION\n')
    for i, v in enumerate(V):
        f.write('{:<4}'.format(i + 1) + ' ' + '{:<4}'.format(v[0]) + ' ' + '{:<4}'.format(v[1]) + '\n')

    f.write('DEMAND_SECTION\n')
    if demandType != 6:
        random.shuffle(D)
    D_with_depot = [0] + D
    for i, _ in enumerate(V):
        f.write('{:<4}'.format(i + 1) + ' ' + '{:<4}'.format(D_with_depot[i]) + '\n')

    f.write('DEPOT_SECTION\n1\n-1\nEOF\n')

print(f"Generated instance: {pathToWrite}")