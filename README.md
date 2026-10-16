# Aerodynamics Simulation

The goal is to simulate airflow around  3D models and visualize aerodynamics through velocity, pressure, vorticity, and surface pressure maps.

**Planned**

* 3D Navier–Stokes solver
* Import `.OBJ` / `.STL` models
* Pressure & velocity visualization
* Surface `Cp` colormapping
* Airflow streamlines / particles
* Drag & lift calculation

Build your own simulations here: https://build.kavang.com/


October 14th, 2026: Progress!

Navier stokes in the functions:
∂u/∂t + (u·∇)u  =  −∇p/ρ  +  ν∇²u          +  f        and   ∇·u = 0
  ^ advect()       ^ project()  ^ diffuse()  ^ forces()      ^ project()

   
<img width="847" height="638" alt="image" src="https://github.com/user-attachments/assets/2f046577-9e56-435f-8b14-f13e87c17257" />

added: velocity advect, pressure, viscosity diffuce. 

Oct 16th, 2026:

<img width="717" height="509" alt="image" src="https://github.com/user-attachments/assets/11505ff0-8931-4077-af3d-7ae50d74d2fa" />

added: multi-shape objects importable by PNG
