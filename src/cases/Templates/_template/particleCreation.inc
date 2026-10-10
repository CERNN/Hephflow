/* ============================================================================
 * PARTICLE CREATION
 * Defines particle properties and initial conditions
 * 
 * This file is included during particle initialization to set up:
 * - Particle positions, velocities, and angular velocities
 * - Particle shapes (sphere, ellipsoid, etc.)
 * - Physical properties (density, diameter, semi-axes)
 * - Collision and movement settings
 * 
 * Available in context:
 *   centerStorage[i]  - Particle center data structure
 *   particles[i]      - Particle object
 *   pShape[i]         - Particle shape
 *   NUM_PARTICLES     - Number of particles (from constants.inc)
 * ========================================================================== */

// Initial position (lattice units)
dfloat3 pos = {(NX-1)/2.0, 65.0, (NZ-1)/2.0};

// Initial velocity (lattice units/timestep)
dfloat3 vel = {0.0, 0.0, 0.0};

// Initial angular velocity (rad/timestep)
dfloat3 w = {0.0, 0.0, 0.0};

// Loop over all particles
for (int i = 0; i < NUM_PARTICLES; i++) {
    ParticleCenter& center = centerStorage[i];

    /* --------------------- PARTICLE GEOMETRY --------------------- */
    // Semi-axes for ellipsoid (for sphere, all equal to radius)
    dfloat3 sa1 = {10.0, 0.0, 0.0};   // X semi-axis
    dfloat3 sa2 = {0.0, 10.0, 0.0};   // Y semi-axis
    dfloat3 sa3 = {0.0, 0.0, 10.0};   // Z semi-axis

    /* --------------------- PARTICLE SHAPE --------------------- */
    // Available shapes: SPHERE, ELLIPSOID
    this->pShape[i] = SPHERE;

    /* --------------------- POSITION AND VELOCITY --------------------- */
    center.setPos(pos);               // Current position
    center.setPos_old(pos);           // Previous position (for time integration)
    center.setVel(vel);               // Current velocity
    center.setVel_old(vel);           // Previous velocity

    /* --------------------- PHYSICAL PROPERTIES --------------------- */
    center.setDensity(1.166666);      // Particle density (ρp/ρf ratio)
    center.setDiameter(30.0);         // Equivalent sphere diameter

    /* --------------------- GEOMETRY SETUP --------------------- */
    center.setSemiAxis1(sa1);
    center.setSemiAxis2(sa2);
    center.setSemiAxis3(sa3);
    center.initializeSemiAxesFromCurrent();

    /* --------------------- MOTION CONTROL --------------------- */
    center.setMovable(true);          // Allow particle to move
    //#define ROTATION_LOCK true      // Uncomment in model.inc to lock rotation

    /* --------------------- ORIENTATION --------------------- */
    center.setQ_pos(dfloat4(0, 0, 0, 1));  // Initial quaternion (no rotation)

    /* --------------------- PARTICLE METHOD --------------------- */
    particles[i].setMethod(IBM);      // Use IBM for fluid coupling
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);

    /* --------------------- COLLISION SETTINGS --------------------- */
    particles[i].setCollideParticle(false);  // Enable/disable particle-particle collision
    particles[i].setCollideWall(true);       // Enable/disable particle-wall collision
}
