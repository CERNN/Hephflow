/**
 *  @file lambda_evolution.cuh
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@gmail.com)
 *  @brief Lambda evolution with model-specific build and break kinetics
 *  @version 0.1.0
 *  @date 02/01/2026
 */

#ifndef __LAMBDA_EVOLUTION_CUH
#define __LAMBDA_EVOLUTION_CUH

#ifdef LAMBDA_DIST

#include "../../non_newtonian/lambda_source.cuh"

/**
 * @brief Compute lambda source term from stress tensor and omega
 * 
 * This function computes the thixotropic source term for lambda evolution
 * based on the local stress state and relaxation parameter.
 * 
 * Usage: Call this function inside kernel where moments are available
 * Variables required: rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_yy_t45, m_zz_t45,
 *                     m_xy_t90, m_xz_t90, m_yz_t90, omegaVar, lambdaVar
 * Produces: lambdaSource (dfloat)
 * 
 * @param nnfProps Fluid properties struct containing thixotropic model info
 * @param rho Density
 * @param ux X-velocity
 * @param uy Y-velocity
 * @param uz Z-velocity
 * @param m_xx XX stress moment
 * @param m_yy YY stress moment
 * @param m_zz ZZ stress moment
 * @param m_xy XY stress moment
 * @param m_xz XZ stress moment
 * @param m_yz YZ stress moment
 * @param omega Relaxation parameter
 * @param lambdaVar Lambda variable (with LAMBDA_ZERO offset)
 * @return Lambda source term
 */
__device__ __forceinline__
dfloat computeLambdaSourceFromStress(
    const fluidProps& nnfProps,
    dfloat rho, dfloat ux, dfloat uy, dfloat uz,
    dfloat m_xx, dfloat m_yy, dfloat m_zz,
    dfloat m_xy, dfloat m_xz, dfloat m_yz,
    dfloat omega, dfloat lambdaVar)
{
    // Compute stress magnitude from moments
    const dfloat S_XX = rho * (m_xx/F_M_II_SCALE - ux*ux/(F_M_I_SCALE*F_M_I_SCALE));
    const dfloat S_YY = rho * (m_yy/F_M_II_SCALE - uy*uy/(F_M_I_SCALE*F_M_I_SCALE));
    const dfloat S_ZZ = rho * (m_zz/F_M_II_SCALE - uz*uz/(F_M_I_SCALE*F_M_I_SCALE));
    const dfloat S_XY = rho * (m_xy/F_M_IJ_SCALE - ux*uy/(F_M_I_SCALE*F_M_I_SCALE));
    const dfloat S_XZ = rho * (m_xz/F_M_IJ_SCALE - ux*uz/(F_M_I_SCALE*F_M_I_SCALE));
    const dfloat S_YZ = rho * (m_yz/F_M_IJ_SCALE - uy*uz/(F_M_I_SCALE*F_M_I_SCALE));

    // Stress magnitude: ||S|| = sqrt(0.5 * S:S)
    const dfloat stressMag = sqrtf(0.5_df * (
        S_XX * S_XX + S_YY * S_YY + S_ZZ * S_ZZ +
        2.0_df * (S_XY * S_XY + S_XZ * S_XZ + S_YZ * S_YZ)));

    // Shear rate: gamma_dot = (1 - 0.5*omega) * ||S|| / eta
    // where eta = (tau - 0.5) / 3
    const dfloat eta = (TAU - 0.5_df) / 3.0_df;
    const dfloat gammaDot = (1.0_df - 0.5_df * omega) * stressMag / eta;

    // Compute source term using model-based dispatch
    return calcLambdaSource(nnfProps, lambdaVar, gammaDot);
}

#endif // LAMBDA_DIST

#endif // __LAMBDA_EVOLUTION_CUH
