# Exact capacitated VRP (single depot, K vehicles, unit demands default)

#### 1. Data declarations ####

set V;                        # all nodes, including depot
param depot symbolic;         # depot node ∈ V
set N := V diff {depot};      # customer nodes

param x{V};                   # x-coordinate
param y{V};                   # y-coordinate

param d{N}  default 1;        # demand at each customer (default = 1)
param Q     > 0;              # vehicle capacity
param K     integer > 0;      # number of vehicles

# Compute Euclidean distances
param c{i in V, j in V} :=
    if i<>j then sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2) else 0;

#### 2. Decision variables ####

# xarc[i,j,k] = 1 if vehicle k travels from i → j
var xarc{i in V, j in V, k in 1..K}, binary;

# MTZ ordering variables: only defined for customers
var u{i in V, k in 1..K} >= 0;

#### 3. Objective: minimize total distance ####

minimize TotalDist:
    sum {k in 1..K, i in V, j in V} c[i,j] * xarc[i,j,k];

#### 4. Constraints ####

# 4.1 Each customer is visited exactly once
s.t. Visit {i in N}:
    sum {k in 1..K, j in V} xarc[i,j,k] = 1;

# 4.2 Flow conservation at customers
s.t. FlowCons {k in 1..K, i in N}:
    sum {j in V} xarc[j,i,k]
  = sum {j in V} xarc[i,j,k];

# 4.3 Each vehicle departs from and returns to the depot exactly once
s.t. DepotOut {k in 1..K}:
    sum {j in N} xarc[depot,j,k] = 1;

s.t. DepotIn  {k in 1..K}:
    sum {i in N} xarc[i,depot,k] = 1;

# 4.4 Capacity constraints
s.t. Capacity {k in 1..K}:
    sum {i in N, j in V} d[i] * xarc[i,j,k] <= Q;

# 4.5 MTZ subtour‐elimination
#   u[depot,k] = 0
s.t. MTZ0 {k in 1..K}:
    u[depot,k] = 0;

#   1 ≤ u[i,k] ≤ |N|  for all customers
s.t. MTZ_bounds {i in N, k in 1..K}:
    1 <= u[i,k] <= card(N);

#   u[i,k] + 1 ≤ u[j,k] + |N|·(1 – xarc[i,j,k])  to eliminate subtours
s.t. MTZ {k in 1..K, i in N, j in N: i <> j}:
    u[i,k] + 1 <= u[j,k] + card(N) * (1 - xarc[i,j,k]);
    
s.t. NoSelfArcs {i in V, k in 1..K}: 
    xarc[i,i,k] = 0;

