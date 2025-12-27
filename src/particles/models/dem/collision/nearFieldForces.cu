#include "nearFieldForces.cuh"

#ifdef ENABLE_LUBRICATION
__device__
void sphereWallLubrication(const CollisionContext& ctx, dfloat gap)
{

    ParticleCenter* pc = ctx.pc_i;

    const dfloat R = pc->getRadius();
    const dfloat h_cut = LUB_CUTOFF_FACTOR * R;
    if (gap > h_cut) return;

    const dfloat h_min = LUB_MIN_FACTOR * R;
    const dfloat h = max(gap, h_min);

    const dfloat3 n = -ctx.wall.normal;

    const dfloat3 v_p = pc->getVel();
    const dfloat3 v_w = ctx.wall.velocity;

    const dfloat3 w_p = pc->getW();

    dfloat3 G = v_p - v_w;

    dfloat un = dot_product(G, n);

    const dfloat mu = (TAU - 0.5) / 3.0;

    const dfloat C = 1.5 * M_PI * mu * R * R;

    dfloat3 f = -(C / h) * (un);

    accumulateForceAndTorque(pc, -f, dfloat3{0,0,0});

}
#endif

#ifdef ENABLE_REPULSIVE_FORCE
__device__
void sphereWallRepulsion(const CollisionContext& ctx, dfloat gap)
{

    ParticleCenter* pc = ctx.pc_i;

    const dfloat R = pc->getRadius();
    const dfloat lambda = REP_RANGE_FACTOR * R;
    if (gap > lambda) return;

    const dfloat3 n = ctx.wall.normal;

    const dfloat F0 = REP_FORCE_AMPLITUDE;

    dfloat force_mag = -F0 * exp(-gap / lambda);
    dfloat3 f = force_mag * n;

    accumulateForceAndTorque(pc, -f, dfloat3{0,0,0});

}
#endif
#ifdef ENABLE_ATTRACTIVE_FORCE
__device__
void sphereWallAttraction(const CollisionContext& ctx, dfloat gap)
{

    ParticleCenter* pc = ctx.pc_i;

    const dfloat R = pc->getRadius();
    const dfloat h_cut = ATT_RANGE_FACTOR * R;
    if (gap > h_cut) return;

    const dfloat h_min = LUB_MIN_FACTOR * R;
    const dfloat h = max(gap, h_min);

    const dfloat3 n = ctx.wall.normal;

    const dfloat A_H = HAMAKER_CONST;

    dfloat force_mag = (A_H * R) / (6.0 * h * h);
    dfloat3 f = force_mag * n;

    accumulateForceAndTorque(pc, -f, dfloat3{0,0,0});

}
#endif

#ifdef ENABLE_LUBRICATION
__device__
void sphereSphereLubrication(const CollisionContext& ctx, dfloat gap)
{

    ParticleCenter* pc_i = ctx.pc_i;
    ParticleCenter* pc_j = ctx.pc_j;

    const dfloat r_i = pc_i->getRadius();
    const dfloat r_j = pc_j->getRadius();
    const dfloat r_eq = (r_i*r_j)/(r_i+r_j);
    const dfloat h_cut = LUB_CUTOFF_FACTOR * r_eq;
    if (gap > h_cut) return;

    const dfloat h_min = LUB_MIN_FACTOR * r_eq;
    const dfloat h = max(gap, h_min);


    dfloat3 n = pc_i->getPos() - pc_j->getPos();
    n = vector_normalize(n);

    dfloat3 G = pc_i->getVel() - pc_j->getVel();
    dfloat un = dot_product(G, n);

    const dfloat mu = (TAU - 0.5) / 3.0;

    const dfloat C = 1.5 * M_PI * mu * r_eq * r_eq;

    dfloat3 f = -(C / h) * (un);

    accumulateForceAndTorque(pc_i, -f, dfloat3{0,0,0});
    accumulateForceAndTorque(pc_j,  f, dfloat3{0,0,0});

}
#endif

#ifdef ENABLE_REPULSIVE_FORCE
__device__
void sphereSphereRepulsion(const CollisionContext& ctx, dfloat gap)
{

    ParticleCenter* pc_i = ctx.pc_i;
    ParticleCenter* pc_j = ctx.pc_j;

    const dfloat r_i = pc_i->getRadius();
    const dfloat r_j = pc_j->getRadius();
    const dfloat r_eq = (r_i*r_j)/(r_i+r_j);
    const dfloat lambda = REP_RANGE_FACTOR * r_eq;
    if (gap > lambda) return;

    const dfloat3 n = pc_i->getPos() - pc_j->getPos();

    const dfloat F0 = REP_FORCE_AMPLITUDE; // lattice units

    dfloat force_mag = -F0 * exp(-gap / lambda);
    dfloat3 f = force_mag * n;

    accumulateForceAndTorque(pc_i, -f, dfloat3{0,0,0});
    accumulateForceAndTorque(pc_j,  f, dfloat3{0,0,0});

}
#endif


#ifdef ENABLE_ATTRACTIVE_FORCE
__device__
void sphereSphereAttraction(const CollisionContext& ctx, dfloat gap)
{

    ParticleCenter* pc_i = ctx.pc_i;
    ParticleCenter* pc_j = ctx.pc_j;

    const dfloat r_i = pc_i->getRadius();
    const dfloat r_j = pc_j->getRadius();
    const dfloat r_eq = (r_i*r_j)/(r_i+r_j);
    const dfloat h_cut = ATT_RANGE_FACTOR * r_eq;
    if (gap > h_cut) return;

    const dfloat h_min = LUB_MIN_FACTOR * r_eq;
    const dfloat h = max(gap, h_min);

    const dfloat3 n = pc_i->getPos() - pc_j->getPos();

    const dfloat A_H = HAMAKER_CONST;

    dfloat force_mag = (A_H * r_eq) / (6.0 * h * h);
    dfloat3 f = force_mag * n;

    accumulateForceAndTorque(pc_i, -f, dfloat3{0,0,0});
    accumulateForceAndTorque(pc_j,  f, dfloat3{0,0,0});

}

#endif