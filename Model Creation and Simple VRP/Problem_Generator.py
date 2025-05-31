#!/usr/bin/env python3
import os
import random
import math
import time
from datetime import datetime
from pathlib import Path
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

# Try both AMPL APIs
try:
    from ampl import AMPL
except ImportError:
    from amplpy import AMPL

def log(msg, start=None):
    """Print a timestamped log message with optional elapsed time."""
    now = datetime.now().strftime('%H:%M:%S')
    if start is not None:
        elapsed = time.time() - start
        print(f"[{now} +{elapsed:.2f}s] {msg}")
    else:
        print(f"[{now}] {msg}")

def generate_data(N, K, script_dir):
    """Generate cheap_problem.dat in script_dir (depot + customers)."""
    coords_range = (0, 100)
    random.seed(42)
    customer_coords = {
        i: (round(random.uniform(*coords_range), 2),
            round(random.uniform(*coords_range), 2))
        for i in range(1, N+1)
    }
    # Depot at centroid
    x0 = round(sum(x for x, y in customer_coords.values()) / N, 2)
    y0 = round(sum(y for x, y in customer_coords.values()) / N, 2)

    lines = ["data;", ""]
    lines.append(f"set N := {' '.join(str(i) for i in range(1, N+1))};")
    lines.append("")
    lines.append("param x :=")
    for i in range(1, N+1):
        lines.append(f"  {i}   {customer_coords[i][0]}")
    lines.append(";")
    lines.append("param y :=")
    for i in range(1, N+1):
        lines.append(f"  {i}   {customer_coords[i][1]}")
    lines.append(";")
    lines.append("")
    lines.append(f"param x0 := {x0};")
    lines.append(f"param y0 := {y0};")
    lines.append("")
    lines.append(f"param K := {K};")
    lines.append("")
    lines.append("param alpha := 2.791;")
    lines.append("param beta  := 0.2669;")
    lines.append("")
    lines.append("end;")

    path = script_dir / 'cheap_problem.dat'
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return str(path)

if __name__ == '__main__':
    total_start = time.time()
    log("Generating data for N=100, K=10")
    script_dir = Path(__file__).resolve().parent
    cheap_dat = generate_data(100, 10, script_dir)
    log("Data generation complete", total_start)

    # Read and solve heuristic model
    cheap_mod = script_dir / 'VRP_Cheap_1.mod'
    ampl = AMPL()
    log(f"Reading model {cheap_mod.name}", total_start)
    ampl.read(str(cheap_mod))
    log(f"Reading data {Path(cheap_dat).name}", total_start)
    ampl.read_data(str(cheap_dat))

    # Use Gurobi with a 2-hour time limit
    ampl.setOption('solver', 'gurobi')
    ampl.setOption('gurobi_options', 'TimeLimit=7200')

    log("Solving heuristic model (Gurobi)", total_start)
    t0 = time.time()
    # Capture stdout/stderr from ampl.solve()
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        ampl.solve()
    solver_output = buf.getvalue()
    duration = time.time() - t0
    obj = ampl.getObjective('TotalCost').value()
    log(f"Heuristic solved in {duration:.2f}s, obj={obj:.6f}", total_start)

    # Capture variable displays
    display_cmds = [
        "display assign;",
        "display cluster_size;",
        "display mu_x; display mu_y;",
        "display stdev_x; display stdev_y;",
        "display bar_c_x; display bar_c_y;",
        "display area_rect;",
        "display g_val;"
    ]
    ampl_output = ampl.getOutput("\n".join(display_cmds))

    # Write combined report
    out_path = script_dir / 'heuristic_solution.txt'
    with open(out_path, 'w') as f:
        f.write("=== Solver Output (Gurobi) ===\n")
        f.write(solver_output)
        f.write("\n\n=== Objective ===\n")
        f.write(f"TotalCost = {obj:.6f}\n")
        f.write("\n=== Variable Displays ===\n")
        f.write(ampl_output)

    print(f"\nSolution and logs written to {out_path}")
