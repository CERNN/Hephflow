// Lambda evolution equation with build and break kinetics
// dlambda/dt = k1*(1-lambda) - k2*lambda*gamma_dot

// Compute shear rate from stress magnitude
#ifdef COMPUTE_SHEAR
    // Stress magnitude: sigma = (1 - omega/2) * auxStressMag
    const dfloat stressMag = (1.0_df - 0.5_df * OMEGA) * auxStressMag;
#else
    // Compute stress from velocity gradients if not already available
    const dfloat S_XX = rhoVar * (m_xx_t45/F_M_II_SCALE - ux_t30*ux_t30/(F_M_I_SCALE*F_M_I_SCALE));
    const dfloat S_YY = rhoVar * (m_yy_t45/F_M_II_SCALE - uy_t30*uy_t30/(F_M_I_SCALE*F_M_I_SCALE));
    const dfloat S_ZZ = rhoVar * (m_zz_t45/F_M_II_SCALE - uz_t30*uz_t30/(F_M_I_SCALE*F_M_I_SCALE));
    const dfloat S_XY = rhoVar * (m_xy_t90/F_M_IJ_SCALE - ux_t30*uy_t30/(F_M_I_SCALE*F_M_I_SCALE));
    const dfloat S_XZ = rhoVar * (m_xz_t90/F_M_IJ_SCALE - ux_t30*uz_t30/(F_M_I_SCALE*F_M_I_SCALE));
    const dfloat S_YZ = rhoVar * (m_yz_t90/F_M_IJ_SCALE - uy_t30*uz_t30/(F_M_I_SCALE*F_M_I_SCALE));
    
    const dfloat stressMag = sqrt(0.5_df * (
        S_XX * S_XX + S_YY * S_YY + S_ZZ * S_ZZ +
        2.0_df * (S_XY * S_XY + S_XZ * S_XZ + S_YZ * S_YZ)));
#endif

// Compute shear rate from stress: gamma_dot = (1 - 0.5*omega) * stress_mag / eta
// where eta = (tau - 0.5) / 3.0
const dfloat eta = (TAU - 0.5_df) / 3.0_df;
const dfloat gammaDot = (1.0_df - 0.5_df * OMEGA) * stressMag / eta;

lambdaVar -= LAMBDA_ZERO;

// Build rate: (1-lambda)/t_build scaled by lambda kinetics constant
dfloat buildRate = LAMBDA_K2 * (1.0_df - lambdaVar);

// Break rate: lambda * gamma_dot / t_break scaled by breakdown kinetics constant
dfloat breakRate = LAMBDA_K1 * lambdaVar * gammaDot;

// Net source term (will be applied after streaming)
lambdaSource = buildRate - breakRate;

lambdaVar += LAMBDA_ZERO;