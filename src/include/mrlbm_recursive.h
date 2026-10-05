#ifndef HEPHFLOW_MRLBM_RECURSIVE_H
#define HEPHFLOW_MRLBM_RECURSIVE_H

// Return the normalized tensor-product Hermite coefficient B_(ax,ay,az)
// reconstructed from velocity and the full second-order Hermite tensor.
// Inputs use Hephflow's stored normalization: 3u, 4.5a_ii, and 9a_ij.
template <int AX, int AY, int AZ, typename T>
__host__ __device__ inline T mrlbmRecursiveCoefficient(
    T ux, T uy, T uz,
    T mxx, T mxy, T mxz, T myy, T myz, T mzz)
{
    constexpr int order = AX + AY + AZ;
    const T oneThird = T(1) / T(3);
    const T u[3] = {ux * oneThird, uy * oneThird, uz * oneThird};
    const T a2[3][3] = {
        {mxx / T(4.5), mxy / T(9), mxz / T(9)},
        {mxy / T(9), myy / T(4.5), myz / T(9)},
        {mxz / T(9), myz / T(9), mzz / T(4.5)}
    };
    int index[6] = {};
    int count = 0;
    for (int i = 0; i < AX; ++i) index[count++] = 0;
    for (int i = 0; i < AY; ++i) index[count++] = 1;
    for (int i = 0; i < AZ; ++i) index[count++] = 2;

    T value = T(1);
    for (int i = 0; i < order; ++i) value *= u[index[i]];
    if (order == 2) value = a2[index[0]][index[1]];
    if (order > 2) {
        for (int p = 0; p < order; ++p) {
            for (int q = p + 1; q < order; ++q) {
                T term = a2[index[p]][index[q]] - u[index[p]] * u[index[q]];
                for (int k = 0; k < order; ++k)
                    if (k != p && k != q) term *= u[index[k]];
                value += term;
            }
        }
    }

    T denominator = T(1);
    if (AX == 2) denominator *= T(2);
    if (AY == 2) denominator *= T(2);
    if (AZ == 2) denominator *= T(2);
    for (int i = 0; i < order; ++i) denominator *= oneThird;
    return value / denominator;
}

// D3Q19 is not a tensor-product quadrature, so continuum Hermites of order
// four and above alias into its hydrodynamic subspace. Project the completed
// reconstruction back onto the requested density, momentum, and second-order
// Hermite tensor while leaving the remaining discrete kinetic content intact.
template <typename T>
__host__ __device__ inline void mrlbmConservativeD3Q19Projection(
    T* pop, T rho, T ux3, T uy3, T uz3,
    T mxx45, T mxy90, T mxz90, T myy45, T myz90, T mzz45)
{
    constexpr int cx[19] = {0,1,-1,0,0,0,0,1,-1,1,-1,0,0,1,-1,1,-1,0,0};
    constexpr int cy[19] = {0,0,0,1,-1,0,0,1,-1,0,0,1,-1,-1,1,0,0,1,-1};
    constexpr int cz[19] = {0,0,0,0,0,1,-1,0,0,1,-1,1,-1,0,0,-1,1,-1,1};
    const T targetA[6] = {rho*mxx45/T(4.5),rho*mxy90/T(9),rho*mxz90/T(9),
                          rho*myy45/T(4.5),rho*myz90/T(9),rho*mzz45/T(4.5)};
    // A second residual-correction pass keeps FP32 recovery within the same
    // tolerance used for individual population comparisons.
    for (int pass=0;pass<2;++pass) {
        T recoveredRho=T(0), recoveredJ[3]={}, recoveredA[6]={};
        for (int p=0;p<19;++p) {
            const T value=pop[p];
            recoveredRho+=value;
            recoveredJ[0]+=value*T(cx[p]); recoveredJ[1]+=value*T(cy[p]); recoveredJ[2]+=value*T(cz[p]);
            recoveredA[0]+=value*(T(cx[p]*cx[p])-T(1)/T(3));
            recoveredA[1]+=value*T(cx[p]*cy[p]); recoveredA[2]+=value*T(cx[p]*cz[p]);
            recoveredA[3]+=value*(T(cy[p]*cy[p])-T(1)/T(3)); recoveredA[4]+=value*T(cy[p]*cz[p]);
            recoveredA[5]+=value*(T(cz[p]*cz[p])-T(1)/T(3));
        }
        const T drho=rho-recoveredRho;
        const T dj[3]={rho*ux3/T(3)-recoveredJ[0],rho*uy3/T(3)-recoveredJ[1],rho*uz3/T(3)-recoveredJ[2]};
        T da[6]; for(int i=0;i<6;++i) da[i]=targetA[i]-recoveredA[i];
        for (int p=0;p<19;++p) {
            const T w=p==0?T(1)/T(3):(p<=6?T(1)/T(18):T(1)/T(36));
            pop[p]+=w*(drho+T(3)*(T(cx[p])*dj[0]+T(cy[p])*dj[1]+T(cz[p])*dj[2])
                +T(4.5)*((T(cx[p]*cx[p])-T(1)/T(3))*da[0]+(T(cy[p]*cy[p])-T(1)/T(3))*da[3]
                         +(T(cz[p]*cz[p])-T(1)/T(3))*da[5])
                +T(9)*(T(cx[p]*cy[p])*da[1]+T(cx[p]*cz[p])*da[2]+T(cy[p]*cz[p])*da[4]));
        }
    }
}

#endif
