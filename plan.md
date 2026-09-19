# 2D CFD Roadmap

Goal: Build a physically accurate 2D incompressible Navier-Stokes simulation.

Note: 100% physical accuracy is impossible with a numerical simulation. The goal is to correctly solve the equations and verify that the solution converges.

## 1. Grid
- Create 2D grid
- Define physical size
- Define dx and dy
- Store velocity
- Store pressure
- Store dye/tracer

## 2. Advection
- Backtrace through velocity
- Add bilinear interpolation
- Advect dye
- Advect velocity
- Test uniform flow

## 3. Viscosity
- Add velocity diffusion
- Implement Laplacian
- Add air viscosity
- Test against known diffusion behavior

## 4. Forces
- Add external forces
- Add controllable wind/inlet
- Verify acceleration behaves correctly

## 5. Pressure
- Calculate velocity divergence
- Solve pressure Poisson equation
- Correct velocity using pressure
- Make divergence approach zero

## 6. Boundaries
- Add solid walls
- Add no-slip boundaries
- Add inlet
- Add outlet
- Add correct pressure boundaries

## 7. Validation
- Test channel flow
- Test lid-driven cavity
- Test vortex decay
- Compare against analytical/reference solutions
- Measure numerical error

## 8. Aerodynamics
- Add square obstacle
- Generate wake
- Calculate vorticity
- Visualize vorticity
- Visualize velocity
- Visualize pressure
- Calculate Reynolds number

## 9. Airfoil
- Add airfoil geometry
- Handle solid boundaries accurately
- Test different angles of attack
- Test different Reynolds numbers
- Calculate lift
- Calculate drag
- Compare with reference data

## 10. Accuracy
- Increase grid resolution
- Decrease timestep
- Check convergence
- Check mass conservation
- Measure numerical diffusion
- Check pressure solver convergence
- Use higher-order numerical methods when needed

## 11. 3D
- Convert grid to 3D
- Implement 3D Navier-Stokes
- Add 3D boundaries
- Validate against known cases
- Increase resolution
- Study vortex dynamics

## Final

Solve:

rho(du/dt + (u·grad)u) = -grad(p) + mu*laplacian(u) + f

with:

div(u) = 0

Use correct physical parameters, boundary conditions, numerical methods, and validation.