# PVWM Planning

Path planning inside a Physically Viable World Model (PVWM). Given a splat scene, the planner finds a collision-free trajectory from a start to a goal pose by solving a spline-based nonlinear program where obstacle costs are computed via Gaussian convolution between the robot body and the scene Gaussians.

## Repository structure

```
foci/
  convolution/   # Gaussian robot-obstacle convolution
  optim/         # Terrain-aware NLP solver (CasADi + IPOPT/HSL)
  planners/      # GroundPlanner, TerrainPlanner
  splines/       # B-spline utilities
  utils/         # PLY splat loader, height-map queries
  visualisation/ # Viser-based 3D viewer
demos/
  hillcounty.py           # Flat-ground planning demo
  hillcounty_terrain.py   # Terrain-following planning demo
  data/
    hillcounty_sm_30000.ply  # Gaussian splat scene (Hill County)
coinhsl/    # HSL sparse linear solvers (required by IPOPT)
wheels/     # Offline Python wheels for isolated environments
```

## Project setup

### 1. Build the image

```bash
docker build -t pvwm-planning .
```

The build compiles HSL (`coinhsl/`), IPOPT, and CasADi from source. Expect ~20–30 minutes on first build.

### 2. Run the container

```bash
docker run --gpus all -it --rm -p 8080:8080 pvwm-planning
docker run -v /home/suann/pvwm-planning:/workspace --gpus all -it --rm -p 8080:8080 pvwm-planning
```

Port `8080` is exposed for the Viser visualiser — open `http://localhost:8080` in your browser once the demo is running.

### 3. Run a demo

Inside the container:

```bash
# Terrain-following planner
python demos/hillcounty_terrain.py

# Flat-ground planner
python demos/hillcounty.py
```

## Local setup (without Docker)

Requires a working IPOPT installation with HSL (see `coinhsl/`) and CasADi built with IPOPT support.

```bash
pip install -e .
pip install --no-deps wheels/*.whl
```

## Dependencies

| Package | Purpose |
|---------|---------|
| CasADi + IPOPT + HSL | Nonlinear trajectory optimisation |
| Viser | 3D visualisation |
| Warp | GPU-accelerated Gaussian queries |
| NumPy / SciPy | Numerics |
| plyfile | Loading `.ply` splat files |
| astar | Initial path guess |
