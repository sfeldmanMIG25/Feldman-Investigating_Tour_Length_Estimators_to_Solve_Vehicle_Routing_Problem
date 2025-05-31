#VRP_Cheap_1.mod

#
# Sets and Parameters
#
set N;                          # customer nodes
param x   {N};                  # x‐coordinate of customer i
param y   {N};                  # y‐coordinate of customer i

param x0;                       # depot x‐coordinate
param y0;                       # depot y‐coordinate

param K integer > 0;            # number of vehicles (fixed by user) - extracted from known optimal solution
set K_veh := 1..K;

# Capacity Data
param d   {N} >= 0;             # demand of customer i
param Q   >= 0;                 # vehicle capacity


# Big‐M bounds for coordinate‐based linearization
param Mx_max   := max {i in N} x[i];
param Mx_min   := min {i in N} x[i];
param Mx_range := Mx_max - Mx_min;

param My_max   := max {i in N} y[i];
param My_min   := min {i in N} y[i];
param My_range := My_max - My_min;

# Regression coefficients (from Çavdar & Sokol, 2015)
param alpha default 2.791;
param beta  default 0.2669;

#
#Decision Variables
#

var assign        {i in N, k in K_veh}, binary;   
    # assign[i,k] = 1 if customer i is assigned to vehicle k

var cluster_size  {k in K_veh}, integer  >= 2;
    # cluster_size[k] = number of nodes in cluster k (customers + depot)

# --- Cluster‐level statistics (including depot) ---
var sum_x   {k in K_veh};
var sum_x2  {k in K_veh};
var sum_y   {k in K_veh};
var sum_y2  {k in K_veh};

var mu_x    {k in K_veh};
var mu_y    {k in K_veh};

var var_x   {k in K_veh} >= 0;
var var_y   {k in K_veh} >= 0;

var stdev_x {k in K_veh} >= 0;
var stdev_y {k in K_veh} >= 0;

var sum_abs_x   {k in K_veh};
var bar_c_x     {k in K_veh};
var sum_abs_x2  {k in K_veh};
var cstdev_x    {k in K_veh} >= 0;

var sum_abs_y   {k in K_veh};
var bar_c_y     {k in K_veh};
var sum_abs_y2  {k in K_veh};
var cstdev_y    {k in K_veh} >= 0;

var x_max   {k in K_veh};
var x_min   {k in K_veh};
var y_max   {k in K_veh};
var y_min   {k in K_veh};

var area_rect {k in K_veh} >= 0;

# --- TSP‐heuristic estimate per cluster ---
var g_val   {k in K_veh} >= 0;

#
# Assignment and Size Constraints
#

# Each customer must be assigned to exactly one vehicle
s.t. OnePerNode {i in N}:
    sum {k in K_veh} assign[i,k] = 1;

# Enforce “balanced” number of customers per route (optional)
s.t. SizeMin {k in K_veh}:
    sum {i in N} assign[i,k] >= floor_size;
s.t. SizeMax {k in K_veh}:
    sum {i in N} assign[i,k] <= ceil_size;

# Define cluster_size = 1 (depot) + (#customers assigned)
s.t. DefClusterSize {k in K_veh}:
    cluster_size[k] = 1 + sum {i in N} assign[i,k];

#
#Capacity Constraints
#

# Total demand on route k cannot exceed vehicle capacity Q
s.t. Capacity {k in K_veh}:
    sum {i in N} d[i] * assign[i,k] <= Q;

#
#Computer Cluster Level Statistics
#

# Sum of x‐coordinates (include depot)
s.t. DefSumX  {k in K_veh}:
    sum_x[k]  = x0 + sum {i in N} x[i] * assign[i,k];

# Sum of squared x‐coordinates (include depot)
s.t. DefSumX2 {k in K_veh}:
    sum_x2[k] = x0^2 + sum {i in N} x[i]^2 * assign[i,k];

# Sum of y‐coordinates (include depot)
s.t. DefSumY  {k in K_veh}:
    sum_y[k]  = y0 + sum {i in N} y[i] * assign[i,k];

# Sum of squared y‐coordinates (include depot)
s.t. DefSumY2 {k in K_veh}:
    sum_y2[k] = y0^2 + sum {i in N} y[i]^2 * assign[i,k];

# Means (μₓ, μᵧ)
s.t. DefMuX {k in K_veh}:
    mu_x[k] * cluster_size[k] = sum_x[k];
s.t. DefMuY {k in K_veh}:
    mu_y[k] * cluster_size[k] = sum_y[k];

# Variances (Varₓ, Varᵧ)
s.t. DefVarX {k in K_veh}:
    var_x[k] = sum_x2[k]/cluster_size[k] - mu_x[k]^2;
s.t. DefVarY {k in K_veh}:
    var_y[k] = sum_y2[k]/cluster_size[k] - mu_y[k]^2;

# Standard deviations (StDevₓ, StDevᵧ)
s.t. DefStdevX {k in K_veh}:
    stdev_x[k]^2 = var_x[k];
s.t. DefStdevY {k in K_veh}:
    stdev_y[k]^2 = var_y[k];

# Mean absolute deviations (MAD) in x & y
s.t. DefSumAbsX {k in K_veh}:
    sum_abs_x[k] = abs(x0 - mu_x[k])
                  + sum {i in N} abs(x[i] - mu_x[k]) * assign[i,k];
s.t. DefBarCX   {k in K_veh}:
    bar_c_x[k] = sum_abs_x[k] / cluster_size[k];

s.t. DefSumAbsY {k in K_veh}:
    sum_abs_y[k] = abs(y0 - mu_y[k])
                  + sum {i in N} abs(y[i] - mu_y[k]) * assign[i,k];
s.t. DefBarCY   {k in K_veh}:
    bar_c_y[k] = sum_abs_y[k] / cluster_size[k];

# Centered standard deviation of absolute deviations (cstdev_x, cstdev_y)
s.t. DefSumAbsX2 {k in K_veh}:
    sum_abs_x2[k] = (abs(x0 - mu_x[k]))^2
                  + sum {i in N} (abs(x[i] - mu_x[k]))^2 * assign[i,k];
s.t. DefCStdevX  {k in K_veh}:
    cstdev_x[k]^2 = sum_abs_x2[k]/cluster_size[k] - bar_c_x[k]^2;

s.t. DefSumAbsY2 {k in K_veh}:
    sum_abs_y2[k] = (abs(y0 - mu_y[k]))^2
                  + sum {i in N} (abs(y[i] - mu_y[k]))^2 * assign[i,k];
s.t. DefCStdevY  {k in K_veh}:
    cstdev_y[k]^2 = sum_abs_y2[k]/cluster_size[k] - bar_c_y[k]^2;

# Axis‐aligned bounding rectangle (x_max, x_min, y_max, y_min)
s.t. DefXMaxCust  {k in K_veh, i in N}:
    x_max[k] >= x[i] - Mx_range * (1 - assign[i,k]);
s.t. DefXMaxDepot {k in K_veh}:
    x_max[k] >= x0;

s.t. DefXMinCust  {k in K_veh, i in N}:
    x_min[k] <= x[i] + Mx_range * (1 - assign[i,k]);
s.t. DefXMinDepot {k in K_veh}:
    x_min[k] <= x0;

s.t. DefYMaxCust  {k in K_veh, i in N}:
    y_max[k] >= y[i] - My_range * (1 - assign[i,k]);
s.t. DefYMaxDepot {k in K_veh}:
    y_max[k] >= y0;

s.t. DefYMinCust  {k in K_veh, i in N}:
    y_min[k] <= y[i] + My_range * (1 - assign[i,k]);
s.t. DefYMinDepot {k in K_veh}:
    y_min[k] <= y0;

s.t. DefArea {k in K_veh}:
    area_rect[k] = (x_max[k] - x_min[k]) * (y_max[k] - y_min[k]);

# 
#TSP‐HEURISTIC (Çavdar–Sokol)
# 

s.t. DefG {k in K_veh}:
    g_val[k]
      = alpha * sqrt( cluster_size[k] * (cstdev_x[k] * cstdev_y[k]) )
      + beta  * sqrt( cluster_size[k] * (stdev_x[k] * stdev_y[k])
                        * (area_rect[k] / (bar_c_x[k] * bar_c_y[k])) );

# 
# Objective
# 
minimize TotalCost:
    sum {k in K_veh} g_val[k];


