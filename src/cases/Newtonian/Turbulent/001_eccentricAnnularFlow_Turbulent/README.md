# Eccentric annular flow

This case models a z-periodic, pressure-driven turbulent flow between a
concave outer cylinder and an offset convex inner cylinder.

Geometry is configured in `constants.inc` with `OUTER_RADIUS`,
`INNER_RADIUS`, and `ECCENTRIC_OFFSET_X/Y`. The default normalized
eccentricity is

```text
sqrt(offset_x^2 + offset_y^2) / (outer_radius - inner_radius) = 0.496
```

Each wall has an independent prescribed rigid velocity:

- `*_TRANSLATION_X/Y/Z` sets uniform translation.
- `*_ANGULAR_VELOCITY` sets rotation about that wall's z axis.

The geometry is static. These values prescribe the kinematic wall velocity;
they do not update cylinder positions during the simulation. Keep transverse
translation at zero for a stationary eccentric geometry. Axial translation
and z-axis rotation are compatible with the periodic, extruded duct.

To build the case, select it in `src/var.h`:

```cpp
#define BC_PROBLEM Newtonian/Turbulent/001_eccentricAnnularFlow_Turbulent
```

Then run `compile.sh` from `src/`. `NODE_TYPE_SAVE` is enabled by default so
the inner and outer boundary-node classifications can be inspected in the
first VTK output.
