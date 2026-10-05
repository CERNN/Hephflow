// ============================================================================
// THERMAL / NUSSELT DIAGNOSTICS
//
// WEST  (x=0)    : T_HOT
// EAST  (x=NX-1) : T_COLD
//
// SOUTH/NORTH/BACK/FRONT : thermally adiabatic
//
// Output files:
//
//   _data_nusselt_x
//   _data_wallnormal_fd
//   _data_wallnormal_moment
//   _data_thermal_state
//
// Nusselt convention:
//
//      Nu = -(Lx / DeltaT) * dT/dx
//
// so both hot and cold wall mean Nu are positive.
//
// ============================================================================

checkCudaErrors(cudaDeviceSynchronize());

checkCudaErrors(cudaMemcpy(
    h_fMom,
    fMom,
    sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS,
    cudaMemcpyDeviceToHost));

checkCudaErrors(cudaDeviceSynchronize());


// ============================================================================
// ACCESSORS
// ============================================================================

auto getT = [&](int x, int y, int z) -> dfloat
{
    return h_fMom[idxMom(
        x % BLOCK_NX,
        y % BLOCK_NY,
        z % BLOCK_NZ,
        M2_C_INDEX,
        x / BLOCK_NX,
        y / BLOCK_NY,
        z / BLOCK_NZ)];
};


auto getMoment = [&](int x, int y, int z, int momentIndex) -> dfloat
{
    return h_fMom[idxMom(
        x % BLOCK_NX,
        y % BLOCK_NY,
        z % BLOCK_NZ,
        momentIndex,
        x / BLOCK_NX,
        y / BLOCK_NY,
        z / BLOCK_NZ)];
};


// ============================================================================
// CONSTANTS
// ============================================================================

// Boundary nodes lie at x=0 and x=NX-1.
//
// Therefore, in lattice units:
//
//      Lx / dx = NX - 1
//
const dfloat L_over_dx =
    static_cast<dfloat>(NX - 1);

const dfloat invDeltaT =
    static_cast<dfloat>(1.0) / T_DELTA_T;

const dfloat tiny =
    static_cast<dfloat>(1.0e-30);


// ============================================================================
// STATISTICS STRUCTURES
// ============================================================================

struct WallNuStats
{
    dfloat sum1   = 0.0;
    dfloat sum2   = 0.0;
    dfloat weight = 0.0;

    dfloat min1 =  1.0e30;
    dfloat max1 = -1.0e30;

    dfloat min2 =  1.0e30;
    dfloat max2 = -1.0e30;

    void add(
        const dfloat nu1,
        const dfloat nu2,
        const dfloat w)
    {
        sum1 += w * nu1;
        sum2 += w * nu2;

        weight += w;

        min1 = std::min(min1, nu1);
        max1 = std::max(max1, nu1);

        min2 = std::min(min2, nu2);
        max2 = std::max(max2, nu2);
    }

    dfloat avg1() const
    {
        return sum1 / weight;
    }

    dfloat avg2() const
    {
        return sum2 / weight;
    }
};


struct AbsStats
{
    dfloat absSum = 0.0;
    dfloat sqSum  = 0.0;
    dfloat maxAbs = 0.0;
    dfloat weight = 0.0;

    void add(
        const dfloat value,
        const dfloat w = 1.0)
    {
        const dfloat a = std::abs(value);

        absSum += w * a;
        sqSum  += w * value * value;
        weight += w;

        maxAbs = std::max(maxAbs, a);
    }

    dfloat meanAbs() const
    {
        return absSum / weight;
    }

    dfloat rms() const
    {
        return std::sqrt(sqSum / weight);
    }
};


// ============================================================================
// 1. HOT / COLD WALL NUSSELT
// ============================================================================

WallNuStats hotFull;
WallNuStats coldFull;

WallNuStats hotInner;
WallNuStats coldInner;


// ----------------------------------------------------------------------------
// Full wall using trapezoidal surface quadrature
// ----------------------------------------------------------------------------

for (int z = 0; z < NZ; ++z)
{
    for (int y = 0; y < NY; ++y)
    {
        const dfloat wy =
            (y == 0 || y == NY - 1) ? 0.5 : 1.0;

        const dfloat wz =
            (z == 0 || z == NZ - 1) ? 0.5 : 1.0;

        const dfloat w = wy * wz;


        // ====================================================================
        // HOT WALL
        // x = 0
        // ====================================================================

        const dfloat Th0 = getT(0, y, z);
        const dfloat Th1 = getT(1, y, z);
        const dfloat Th2 = getT(2, y, z);


        const dfloat dThdx1 =
            Th1 - Th0;

        const dfloat dThdx2 =
            (-3.0 * Th0
             +4.0 * Th1
             -      Th2) / 2.0;


        const dfloat NuHot1 =
            -dThdx1 * L_over_dx * invDeltaT;

        const dfloat NuHot2 =
            -dThdx2 * L_over_dx * invDeltaT;


        hotFull.add(
            NuHot1,
            NuHot2,
            w);


        // ====================================================================
        // COLD WALL
        // x = NX-1
        //
        // Derivative remains expressed in global +x direction.
        // ====================================================================

        const dfloat Tc0 = getT(NX - 1, y, z);
        const dfloat Tc1 = getT(NX - 2, y, z);
        const dfloat Tc2 = getT(NX - 3, y, z);


        const dfloat dTcdx1 =
            Tc0 - Tc1;

        const dfloat dTcdx2 =
            ( 3.0 * Tc0
             -4.0 * Tc1
             +      Tc2) / 2.0;


        const dfloat NuCold1 =
            -dTcdx1 * L_over_dx * invDeltaT;

        const dfloat NuCold2 =
            -dTcdx2 * L_over_dx * invDeltaT;


        coldFull.add(
            NuCold1,
            NuCold2,
            w);
    }
}


// ----------------------------------------------------------------------------
// Interior-only wall diagnostic
//
// Remove:
//
// y = 0, NY-1
// z = 0, NZ-1
//
// This allows us to see whether wall intersections dominate the Nu imbalance.
// ----------------------------------------------------------------------------

for (int z = 1; z < NZ - 1; ++z)
{
    for (int y = 1; y < NY - 1; ++y)
    {
        // HOT

        const dfloat Th0 = getT(0, y, z);
        const dfloat Th1 = getT(1, y, z);
        const dfloat Th2 = getT(2, y, z);


        const dfloat dThdx1 =
            Th1 - Th0;

        const dfloat dThdx2 =
            (-3.0 * Th0
             +4.0 * Th1
             -      Th2) / 2.0;


        const dfloat NuHot1 =
            -dThdx1 * L_over_dx * invDeltaT;

        const dfloat NuHot2 =
            -dThdx2 * L_over_dx * invDeltaT;


        hotInner.add(
            NuHot1,
            NuHot2,
            1.0);


        // COLD

        const dfloat Tc0 = getT(NX - 1, y, z);
        const dfloat Tc1 = getT(NX - 2, y, z);
        const dfloat Tc2 = getT(NX - 3, y, z);


        const dfloat dTcdx1 =
            Tc0 - Tc1;

        const dfloat dTcdx2 =
            ( 3.0 * Tc0
             -4.0 * Tc1
             +      Tc2) / 2.0;


        const dfloat NuCold1 =
            -dTcdx1 * L_over_dx * invDeltaT;

        const dfloat NuCold2 =
            -dTcdx2 * L_over_dx * invDeltaT;


        coldInner.add(
            NuCold1,
            NuCold2,
            1.0);
    }
}


// ============================================================================
// HOT / COLD HEAT-BALANCE ERROR
// ============================================================================

const dfloat NuBalance1Full =
    std::abs(hotFull.avg1() - coldFull.avg1())
    /
    (
        0.5 *
        (
            std::abs(hotFull.avg1())
            +
            std::abs(coldFull.avg1())
        )
        + tiny
    );


const dfloat NuBalance2Full =
    std::abs(hotFull.avg2() - coldFull.avg2())
    /
    (
        0.5 *
        (
            std::abs(hotFull.avg2())
            +
            std::abs(coldFull.avg2())
        )
        + tiny
    );


const dfloat NuBalance1Inner =
    std::abs(hotInner.avg1() - coldInner.avg1())
    /
    (
        0.5 *
        (
            std::abs(hotInner.avg1())
            +
            std::abs(coldInner.avg1())
        )
        + tiny
    );


const dfloat NuBalance2Inner =
    std::abs(hotInner.avg2() - coldInner.avg2())
    /
    (
        0.5 *
        (
            std::abs(hotInner.avg2())
            +
            std::abs(coldInner.avg2())
        )
        + tiny
    );


// ============================================================================
// 2. MACROSCOPIC WALL-NORMAL FD GRADIENT
//
// IMPORTANT:
//
// This is NOT called "heat leakage" anymore.
//
// It measures the wall-normal gradient inferred from T0,T1,T2.
//
// For an LBM zero-flux wall this may not be identically zero, especially at
// a moving lid / strong thermal boundary layer.
//
// Edges are excluded so that only pure face nodes are tested.
// ============================================================================

AbsStats fdY0_1;
AbsStats fdY0_2;

AbsStats fdY1_1;
AbsStats fdY1_2;

AbsStats fdZ0_1;
AbsStats fdZ0_2;

AbsStats fdZ1_1;
AbsStats fdZ1_2;


// ----------------------------------------------------------------------------
// SOUTH / NORTH
// ----------------------------------------------------------------------------

for (int z = 1; z < NZ - 1; ++z)
{
    for (int x = 1; x < NX - 1; ++x)
    {
        // SOUTH y=0

        const dfloat Ts0 = getT(x, 0, z);
        const dfloat Ts1 = getT(x, 1, z);
        const dfloat Ts2 = getT(x, 2, z);


        const dfloat dTdy1_s =
            Ts1 - Ts0;

        const dfloat dTdy2_s =
            (-3.0 * Ts0
             +4.0 * Ts1
             -      Ts2) / 2.0;


        // Nusselt-like normalization only for convenient comparison
        // with the x-wall heat-transfer scale.

        const dfloat gradNu1_s =
            dTdy1_s * L_over_dx * invDeltaT;

        const dfloat gradNu2_s =
            dTdy2_s * L_over_dx * invDeltaT;


        fdY0_1.add(gradNu1_s);
        fdY0_2.add(gradNu2_s);


        // NORTH y=NY-1
        //
        // global +y derivative

        const dfloat Tn0 = getT(x, NY - 1, z);
        const dfloat Tn1 = getT(x, NY - 2, z);
        const dfloat Tn2 = getT(x, NY - 3, z);


        const dfloat dTdy1_n =
            Tn0 - Tn1;

        const dfloat dTdy2_n =
            ( 3.0 * Tn0
             -4.0 * Tn1
             +      Tn2) / 2.0;


        const dfloat gradNu1_n =
            dTdy1_n * L_over_dx * invDeltaT;

        const dfloat gradNu2_n =
            dTdy2_n * L_over_dx * invDeltaT;


        fdY1_1.add(gradNu1_n);
        fdY1_2.add(gradNu2_n);
    }
}


// ----------------------------------------------------------------------------
// BACK / FRONT
// ----------------------------------------------------------------------------

for (int y = 1; y < NY - 1; ++y)
{
    for (int x = 1; x < NX - 1; ++x)
    {
        // BACK z=0

        const dfloat Tb0 = getT(x, y, 0);
        const dfloat Tb1 = getT(x, y, 1);
        const dfloat Tb2 = getT(x, y, 2);


        const dfloat dTdz1_b =
            Tb1 - Tb0;

        const dfloat dTdz2_b =
            (-3.0 * Tb0
             +4.0 * Tb1
             -      Tb2) / 2.0;


        const dfloat gradNu1_b =
            dTdz1_b * L_over_dx * invDeltaT;

        const dfloat gradNu2_b =
            dTdz2_b * L_over_dx * invDeltaT;


        fdZ0_1.add(gradNu1_b);
        fdZ0_2.add(gradNu2_b);


        // FRONT z=NZ-1

        const dfloat Tf0 = getT(x, y, NZ - 1);
        const dfloat Tf1 = getT(x, y, NZ - 2);
        const dfloat Tf2 = getT(x, y, NZ - 3);


        const dfloat dTdz1_f =
            Tf0 - Tf1;

        const dfloat dTdz2_f =
            ( 3.0 * Tf0
             -4.0 * Tf1
             +      Tf2) / 2.0;


        const dfloat gradNu1_f =
            dTdz1_f * L_over_dx * invDeltaT;

        const dfloat gradNu2_f =
            dTdz2_f * L_over_dx * invDeltaT;


        fdZ1_1.add(gradNu1_f);
        fdZ1_2.add(gradNu2_f);
    }
}


// ============================================================================
// 3. DIRECT LBM NORMAL SCALAR MOMENT
//
// This is the more direct diagnostic of the population zero-flux BC.
//
// Stored first moments have already been scaled by F_M_I_SCALE in your
// current kernel, so undo that scaling here:
//
//      J = stored_M2_C? / F_M_I_SCALE
//
// For:
//   SOUTH/NORTH -> Jy
//   BACK/FRONT   -> Jz
//
// At a no-penetration adiabatic wall these should be essentially zero.
//
// Again, edges/corners are excluded.
// ============================================================================

AbsStats JySouth;
AbsStats JyNorth;

AbsStats JzBack;
AbsStats JzFront;


for (int z = 1; z < NZ - 1; ++z)
{
    for (int x = 1; x < NX - 1; ++x)
    {
        const dfloat Jy_south =
            getMoment(
                x,
                0,
                z,
                M2_CY_INDEX)
            / F_M_I_SCALE;


        const dfloat Jy_north =
            getMoment(
                x,
                NY - 1,
                z,
                M2_CY_INDEX)
            / F_M_I_SCALE;


        JySouth.add(Jy_south);
        JyNorth.add(Jy_north);
    }
}


for (int y = 1; y < NY - 1; ++y)
{
    for (int x = 1; x < NX - 1; ++x)
    {
        const dfloat Jz_back =
            getMoment(
                x,
                y,
                0,
                M2_CZ_INDEX)
            / F_M_I_SCALE;


        const dfloat Jz_front =
            getMoment(
                x,
                y,
                NZ - 1,
                M2_CZ_INDEX)
            / F_M_I_SCALE;


        JzBack.add(Jz_back);
        JzFront.add(Jz_front);
    }
}


// ============================================================================
// 4. THERMAL STATE
// ============================================================================

dfloat maxHotTempError  = 0.0;
dfloat maxColdTempError = 0.0;

dfloat meanHotTempError  = 0.0;
dfloat meanColdTempError = 0.0;

dfloat wallWeight = 0.0;


// ----------------------------------------------------------------------------
// Dirichlet error
// ----------------------------------------------------------------------------

for (int z = 0; z < NZ; ++z)
{
    for (int y = 0; y < NY; ++y)
    {
        const dfloat wy =
            (y == 0 || y == NY - 1) ? 0.5 : 1.0;

        const dfloat wz =
            (z == 0 || z == NZ - 1) ? 0.5 : 1.0;

        const dfloat w =
            wy * wz;


        const dfloat eHot =
            std::abs(
                getT(0, y, z)
                - T_HOT);


        const dfloat eCold =
            std::abs(
                getT(NX - 1, y, z)
                - T_COLD);


        meanHotTempError +=
            w * eHot;

        meanColdTempError +=
            w * eCold;


        maxHotTempError =
            std::max(
                maxHotTempError,
                eHot);


        maxColdTempError =
            std::max(
                maxColdTempError,
                eCold);


        wallWeight += w;
    }
}


meanHotTempError /=
    wallWeight;

meanColdTempError /=
    wallWeight;


// ----------------------------------------------------------------------------
// Volume-integrated mean T + min/max
// ----------------------------------------------------------------------------

dfloat TvolumeSum =
    0.0;

dfloat volumeWeight =
    0.0;

dfloat Tmin =
    1.0e30;

dfloat Tmax =
    -1.0e30;


for (int z = 0; z < NZ; ++z)
{
    const dfloat wz =
        (z == 0 || z == NZ - 1)
        ? 0.5
        : 1.0;

    for (int y = 0; y < NY; ++y)
    {
        const dfloat wy =
            (y == 0 || y == NY - 1)
            ? 0.5
            : 1.0;

        for (int x = 0; x < NX; ++x)
        {
            const dfloat wx =
                (x == 0 || x == NX - 1)
                ? 0.5
                : 1.0;


            const dfloat w =
                wx * wy * wz;


            const dfloat T =
                getT(x, y, z);


            TvolumeSum +=
                w * T;

            volumeWeight +=
                w;


            Tmin =
                std::min(Tmin, T);

            Tmax =
                std::max(Tmax, T);
        }
    }
}


const dfloat Tmean =
    TvolumeSum / volumeWeight;


const dfloat thetaMean =
    (Tmean - T_COLD)
    /
    T_DELTA_T;


// ============================================================================
// THERMAL STEADY-STATE RATE
//
// d(thetaMean)/d(step)
//
// First sample is written as zero.
// ============================================================================

static bool thetaHistoryInitialized =
    false;

static dfloat previousThetaMean =
    0.0;

static long long previousThetaStep =
    0;


dfloat dThetaMean_dStep =
    0.0;


if (thetaHistoryInitialized)
{
    const long long dStep =
        static_cast<long long>(step)
        - previousThetaStep;


    if (dStep != 0)
    {
        dThetaMean_dStep =
            (thetaMean - previousThetaMean)
            /
            static_cast<dfloat>(dStep);
    }
}


previousThetaMean =
    thetaMean;

previousThetaStep =
    static_cast<long long>(step);

thetaHistoryInitialized =
    true;


// ============================================================================
// HEADERS
// ============================================================================

if (step == REPORT_SAVE)
{
    // ------------------------------------------------------------------------
    // NUSSELT
    // ------------------------------------------------------------------------

    saveTreatDataHeader(
        "_data_nusselt_x",

        "step,"
        "NuHot2_full,NuCold2_full,Balance2_full,"
        "NuHot1_full,NuCold1_full,Balance1_full,"
        "NuHot2_inner,NuCold2_inner,Balance2_inner,"
        "NuHot1_inner,NuCold1_inner,Balance1_inner,"
        "NuHot2_min,NuHot2_max,"
        "NuCold2_min,NuCold2_max,"
        "NuHot1_min,NuHot1_max,"
        "NuCold1_min,NuCold1_max"
    );


    // ------------------------------------------------------------------------
    // MACROSCOPIC FD WALL-NORMAL GRADIENT
    // ------------------------------------------------------------------------

    saveTreatDataHeader(
        "_data_wallnormal_fd",

        "step,"
        "South2_meanAbs,South2_RMS,South2_maxAbs,"
        "North2_meanAbs,North2_RMS,North2_maxAbs,"
        "Back2_meanAbs,Back2_RMS,Back2_maxAbs,"
        "Front2_meanAbs,Front2_RMS,Front2_maxAbs,"
        "South1_meanAbs,South1_RMS,South1_maxAbs,"
        "North1_meanAbs,North1_RMS,North1_maxAbs,"
        "Back1_meanAbs,Back1_RMS,Back1_maxAbs,"
        "Front1_meanAbs,Front1_RMS,Front1_maxAbs"
    );


    // ------------------------------------------------------------------------
    // DIRECT NORMAL SCALAR MOMENT
    // ------------------------------------------------------------------------

    saveTreatDataHeader(
        "_data_wallnormal_moment",

        "step,"
        "South_Jy_meanAbs,South_Jy_RMS,South_Jy_maxAbs,"
        "North_Jy_meanAbs,North_Jy_RMS,North_Jy_maxAbs,"
        "Back_Jz_meanAbs,Back_Jz_RMS,Back_Jz_maxAbs,"
        "Front_Jz_meanAbs,Front_Jz_RMS,Front_Jz_maxAbs"
    );


    // ------------------------------------------------------------------------
    // THERMAL STATE
    // ------------------------------------------------------------------------

    saveTreatDataHeader(
        "_data_thermal_state",

        "step,"
        "Tmean,thetaMean,dThetaMean_dStep,"
        "Tmin,Tmax,"
        "HotT_meanError,HotT_maxError,"
        "ColdT_meanError,ColdT_maxError"
    );
}


// ============================================================================
// SAVE NUSSELT
// ============================================================================

{
    std::ostringstream ss;

    ss << std::scientific
       << std::setprecision(10)

       << step << ","

       // SECOND ORDER - FULL WALL
       << hotFull.avg2() << ","
       << coldFull.avg2() << ","
       << NuBalance2Full << ","

       // FIRST ORDER - FULL WALL
       << hotFull.avg1() << ","
       << coldFull.avg1() << ","
       << NuBalance1Full << ","

       // SECOND ORDER - INNER WALL
       << hotInner.avg2() << ","
       << coldInner.avg2() << ","
       << NuBalance2Inner << ","

       // FIRST ORDER - INNER WALL
       << hotInner.avg1() << ","
       << coldInner.avg1() << ","
       << NuBalance1Inner << ","

       // SECOND ORDER EXTREMA
       << hotFull.min2 << ","
       << hotFull.max2 << ","
       << coldFull.min2 << ","
       << coldFull.max2 << ","

       // FIRST ORDER EXTREMA
       << hotFull.min1 << ","
       << hotFull.max1 << ","
       << coldFull.min1 << ","
       << coldFull.max1;


    saveTreatData(
        "_data_nusselt_x",
        ss.str(),
        step,
        true);
}


// ============================================================================
// SAVE MACROSCOPIC FD WALL-NORMAL GRADIENT
// ============================================================================

{
    std::ostringstream ss;

    ss << std::scientific
       << std::setprecision(10)

       << step << ","

       // SECOND ORDER
       << fdY0_2.meanAbs() << ","
       << fdY0_2.rms() << ","
       << fdY0_2.maxAbs << ","

       << fdY1_2.meanAbs() << ","
       << fdY1_2.rms() << ","
       << fdY1_2.maxAbs << ","

       << fdZ0_2.meanAbs() << ","
       << fdZ0_2.rms() << ","
       << fdZ0_2.maxAbs << ","

       << fdZ1_2.meanAbs() << ","
       << fdZ1_2.rms() << ","
       << fdZ1_2.maxAbs << ","

       // FIRST ORDER
       << fdY0_1.meanAbs() << ","
       << fdY0_1.rms() << ","
       << fdY0_1.maxAbs << ","

       << fdY1_1.meanAbs() << ","
       << fdY1_1.rms() << ","
       << fdY1_1.maxAbs << ","

       << fdZ0_1.meanAbs() << ","
       << fdZ0_1.rms() << ","
       << fdZ0_1.maxAbs << ","

       << fdZ1_1.meanAbs() << ","
       << fdZ1_1.rms() << ","
       << fdZ1_1.maxAbs;


    saveTreatData(
        "_data_wallnormal_fd",
        ss.str(),
        step,
        true);
}


// ============================================================================
// SAVE DIRECT NORMAL SCALAR MOMENT
// ============================================================================

{
    std::ostringstream ss;

    ss << std::scientific
       << std::setprecision(10)

       << step << ","

       << JySouth.meanAbs() << ","
       << JySouth.rms() << ","
       << JySouth.maxAbs << ","

       << JyNorth.meanAbs() << ","
       << JyNorth.rms() << ","
       << JyNorth.maxAbs << ","

       << JzBack.meanAbs() << ","
       << JzBack.rms() << ","
       << JzBack.maxAbs << ","

       << JzFront.meanAbs() << ","
       << JzFront.rms() << ","
       << JzFront.maxAbs;


    saveTreatData(
        "_data_wallnormal_moment",
        ss.str(),
        step,
        true);
}


// ============================================================================
// SAVE THERMAL STATE
// ============================================================================

{
    std::ostringstream ss;

    ss << std::scientific
       << std::setprecision(10)

       << step << ","

       << Tmean << ","
       << thetaMean << ","
       << dThetaMean_dStep << ","

       << Tmin << ","
       << Tmax << ","

       << meanHotTempError << ","
       << maxHotTempError << ","

       << meanColdTempError << ","
       << maxColdTempError;


    saveTreatData(
        "_data_thermal_state",
        ss.str(),
        step,
        true);
}