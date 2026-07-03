#include "treatData.cuh"

__host__
void treatData(const TreatDataParams* params)
{
    // Unpack parameters from struct
    dfloat* h_fMom = params->h_fMom;
    dfloat* fMom = params->d_fMom;
    #if MEAN_FLOW
    dfloat* fMom_mean = params->d_fMom_mean;
    #endif
    #ifdef BC_FORCES
    dfloat* d_BC_Fx = params->d_BC_Fx;
    dfloat* d_BC_Fy = params->d_BC_Fy;
    dfloat* d_BC_Fz = params->d_BC_Fz;
    #endif
    unsigned int step = params->step;
    size_t zOffset = params->zOffset;

    #ifdef TREAT_DATA_INCLUDE
    // #include CASE_TREAT_DATA
    copyMacroscopic(h_fMom, fMom, step, zOffset);
    #endif //TREAT_DATA_INCLUDE

    //totalKineticEnergy(fMom,step);         
}

__host__
void mean_moment(dfloat *fMom, dfloat *meanMom, int m_index, size_t step, int target){

    dfloat* sum;
    cudaMalloc((void**)&sum, NUM_BLOCK_LOCAL * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = (NZ/N_GPUS) / nt_z;

    sumReductionThread << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z_LOCAL), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom, sum,m_index);

    nb_x = NUM_BLOCK_X;
    nb_y = NUM_BLOCK_Y;
    nb_z = NUM_BLOCK_Z_LOCAL;

    int current_block_size = nb_x * nb_y * nb_z;
   
    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sum, sum);
            break;
        }
        else {
            nb_x = (nb_x < BLOCK_NX ? 1 : nb_x / BLOCK_NX);
            nb_y = (nb_y < BLOCK_NY ? 1 : nb_y / BLOCK_NY);
            nb_z = (nb_z < BLOCK_NZ ? 1 : nb_z / BLOCK_NZ);
            if (nb_x * nb_y * nb_z * nt_x * nt_y * nt_z > current_block_size) {
                if (nb_x > nb_y && nb_x > nb_z)
                    nt_x /= 2;
                else if (nb_y > nb_x && nb_y > nb_z)
                    nt_y /= 2;
                else
                    nt_z /= 2;
            }
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sum, sum);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp;
    
    checkCudaErrors(cudaMemcpy(&temp, sum, sizeof(dfloat), cudaMemcpyDeviceToHost)); 

    if (m_index == M_RHO_INDEX){
        temp = (temp/(dfloat)NUMBER_LBM_NODES_LOCAL); 
    }else{
        temp = (temp/(dfloat)NUMBER_LBM_NODES_LOCAL);
    }
                
    if (target == 0){
        checkCudaErrors(cudaMemcpy(meanMom, &temp, sizeof(dfloat), cudaMemcpyHostToDevice)); 
    }
    else{
        checkCudaErrors(cudaMemcpy(meanMom, &temp, sizeof(dfloat), cudaMemcpyHostToHost)); 
    }


    cudaFree(sum);


    
}

__host__ 
void totalKineticEnergy(
    dfloat *fMom, 
    size_t step
){
    dfloat* sumKE;
    cudaMalloc((void**)&sumKE, NUM_BLOCK_LOCAL * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = (NZ/N_GPUS) / nt_z;

    sumReductionThread_KE << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z_LOCAL), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom, sumKE);

    nb_x = NUM_BLOCK_X;
    nb_y = NUM_BLOCK_Y;
    nb_z = NUM_BLOCK_Z_LOCAL;

    int current_block_size = nb_x * nb_y * nb_z;

    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sumKE, sumKE);
            break;
        }
        else {
            nb_x = (nb_x < BLOCK_NX ? 1 : nb_x / BLOCK_NX);
            nb_y = (nb_y < BLOCK_NY ? 1 : nb_y / BLOCK_NY);
            nb_z = (nb_z < BLOCK_NZ ? 1 : nb_z / BLOCK_NZ);
            if (nb_x * nb_y * nb_z * nt_x * nt_y * nt_z > current_block_size) {
                if (nb_x > nb_y && nb_x > nb_z)
                    nt_x /= 2;
                else if (nb_y > nb_x && nb_y > nb_z)
                    nt_y /= 2;
                else
                    nt_z /= 2;
            }
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sumKE, sumKE);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp;
    
    checkCudaErrors(cudaMemcpy(&temp, sumKE, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    temp = (temp)/(NUMBER_LBM_NODES_LOCAL);

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    strDataInfo <<"step,"<< step<< "," << temp;// << "," << mean_counter;



    saveTreatData("_totalKineticEnergy",strDataInfo.str(),step);
    cudaFree(sumKE);
}

#ifdef CONVECTION_DIFFUSION_TRANSPORT
#ifdef CONFORMATION_TENSOR
__host__ 
void totalSpringEnergy(
    dfloat *fMom, 
    size_t step
){
    dfloat* sumKE;
    cudaMalloc((void**)&sumKE, NUM_BLOCK_LOCAL * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = (NZ/N_GPUS) / nt_z;

    sumReductionThread_SE << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z_LOCAL), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom, sumKE);

    nb_x = NUM_BLOCK_X;
    nb_y = NUM_BLOCK_Y;
    nb_z = NUM_BLOCK_Z_LOCAL;

    int current_block_size = nb_x * nb_y * nb_z;

    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sumKE, sumKE);
            break;
        }
        else {
            nb_x = (nb_x < BLOCK_NX ? 1 : nb_x / BLOCK_NX);
            nb_y = (nb_y < BLOCK_NY ? 1 : nb_y / BLOCK_NY);
            nb_z = (nb_z < BLOCK_NZ ? 1 : nb_z / BLOCK_NZ);
            if (nb_x * nb_y * nb_z * nt_x * nt_y * nt_z > current_block_size) {
                if (nb_x > nb_y && nb_x > nb_z)
                    nt_x /= 2;
                else if (nb_y > nb_x && nb_y > nb_z)
                    nt_y /= 2;
                else
                    nt_z /= 2;
            }
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sumKE, sumKE);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp;
    
    checkCudaErrors(cudaMemcpy(&temp, sumKE, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    temp = (temp/2.0_df) * nu_p * inv_lambda;
    temp = (temp)/(NUMBER_LBM_NODES_LOCAL);

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    strDataInfo <<"step,"<< step<< "," << temp;// << "," << mean_counter;



    saveTreatData("_totalSpringEnergy",strDataInfo.str(),step);
    cudaFree(sumKE);
}
#endif //CONFORMATION_TENSOR
#endif //CONVECTION_DIFFUSION_TRANSPORT

__host__ 
void turbulentKineticEnergy(
    dfloat *fMom, 
    dfloat *m_fMom, 
    size_t step
){

    dfloat* sumTKE;
    cudaMalloc((void**)&sumTKE, NUM_BLOCK_LOCAL * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = (NZ/N_GPUS) / nt_z;

    sumReductionThread_TKE << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z_LOCAL), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom,m_fMom,sumTKE);

    int current_block_size = nb_x * nb_y * nb_z;

    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sumTKE, sumTKE);
            break;
        }
        else {
            nb_x = (nb_x < BLOCK_NX ? 1 : nb_x / BLOCK_NX);
            nb_y = (nb_y < BLOCK_NY ? 1 : nb_y / BLOCK_NY);
            nb_z = (nb_z < BLOCK_NZ ? 1 : nb_z / BLOCK_NZ);
            if (nb_x * nb_y * nb_z * nt_x * nt_y * nt_z > current_block_size) {
                if (nb_x > nb_y && nb_x > nb_z)
                    nt_x /= 2;
                else if (nb_y > nb_x && nb_y > nb_z)
                    nt_y /= 2;
                else
                    nt_z /= 2;
            }
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sumTKE, sumTKE);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp;
    
    checkCudaErrors(cudaMemcpy(&temp, sumTKE, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    temp = (temp)/(U_MAX*U_MAX*NUMBER_LBM_NODES_LOCAL);

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    strDataInfo <<"step,"<< step<< "," << temp;// << "," << mean_counter;



    saveTreatData("_turbulentKineticEnergy",strDataInfo.str(),step);
    cudaFree(sumTKE);

}


void totalBcDrag(
    dfloat *d_BC_Fx, 
    dfloat* d_BC_Fy, 
    dfloat* d_BC_Fz, 
    size_t step
){
    dfloat* sum_BC_Fx;
    dfloat* sum_BC_Fy;
    dfloat* sum_BC_Fz;

    cudaMalloc((void**)&sum_BC_Fx, NUM_BLOCK_LOCAL * sizeof(dfloat));
    cudaMalloc((void**)&sum_BC_Fy, NUM_BLOCK_LOCAL * sizeof(dfloat));
    cudaMalloc((void**)&sum_BC_Fz, NUM_BLOCK_LOCAL * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = (NZ/N_GPUS) / nt_z;

    sumReductionScalar << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z_LOCAL), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (d_BC_Fx, sum_BC_Fx);
    sumReductionScalar << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z_LOCAL), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (d_BC_Fy, sum_BC_Fy);
    sumReductionScalar << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z_LOCAL), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (d_BC_Fz, sum_BC_Fz);

    nb_x = NUM_BLOCK_X;
    nb_y = NUM_BLOCK_Y;
    nb_z = NUM_BLOCK_Z_LOCAL;

    int current_block_size = nb_x * nb_y * nb_z;

    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sum_BC_Fx, sum_BC_Fx);
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sum_BC_Fy, sum_BC_Fy);
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sum_BC_Fz, sum_BC_Fz);
            break;
        }
        else {
            nb_x = (nb_x < BLOCK_NX ? 1 : nb_x / BLOCK_NX);
            nb_y = (nb_y < BLOCK_NY ? 1 : nb_y / BLOCK_NY);
            nb_z = (nb_z < BLOCK_NZ ? 1 : nb_z / BLOCK_NZ);
            if (nb_x * nb_y * nb_z * nt_x * nt_y * nt_z > current_block_size) {
                if (nb_x > nb_y && nb_x > nb_z)
                    nt_x /= 2;
                else if (nb_y > nb_x && nb_y > nb_z)
                    nt_y /= 2;
                else
                    nt_z /= 2;
            }
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sum_BC_Fx, sum_BC_Fx);
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sum_BC_Fy, sum_BC_Fy);
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sum_BC_Fz, sum_BC_Fz);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp_x, temp_y, temp_z;
    
    checkCudaErrors(cudaMemcpy(&temp_x, sum_BC_Fx, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&temp_y, sum_BC_Fy, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&temp_z, sum_BC_Fz, sizeof(dfloat), cudaMemcpyDeviceToHost)); 


    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    strDataInfo <<"step,"<< step<< "," << temp_x<< "," << temp_y<< "," << temp_z;// << "," << mean_counter;

    saveTreatData("_totalBcDrag",strDataInfo.str(),step);

    cudaFree(sum_BC_Fx);
    cudaFree(sum_BC_Fy);
    cudaFree(sum_BC_Fz);


};

__host__
void rhoProfile(
    dfloat* fMom,
    int dir_index,
    int x0, int y0, int z0,
    unsigned int step
){
    std::ostringstream strDataInfo;
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);
    strDataInfo << "step " << step;

    int x_loc, y_loc, z_loc;
    dfloat hostVal; // use a stack host variable for single-element copies
    std::stringstream name;

    switch (dir_index)
    {
    case 1: // rho on x-direction
        y_loc = y0;
        z_loc = z0;
        for (x_loc = 0; x_loc < NX; ++x_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             0, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << hostVal;
        }
        name << "rhoProfile_dx_y" << y0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 2: // rho on y-direction
        x_loc = x0;
        z_loc = z0;
        for (y_loc = 0; y_loc < NY; ++y_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             0, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            // optional: validate idx if you have TOTAL_SIZE available
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << hostVal;
        }
        name << "rhoProfile_dy_x" << x0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 3: // rho on z-direction
        y_loc = y0;
        x_loc = x0;
        for (z_loc = 0; z_loc < NZ_TOTAL; ++z_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             0, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << hostVal;
        }
        name << "rhoProfile_dz_x" << x0 << "_y" << y0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    default:
        std::cerr << "rhoProfile: unknown dir_index " << dir_index << std::endl;
        break;
    }
}

__host__
void velocityProfile(
    dfloat* fMom,
    int dir_index,
    int x0, int y0, int z0,
    unsigned int step
){
    std::ostringstream strDataInfo;
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);
    strDataInfo << "step " << step;

    int x_loc, y_loc, z_loc;
    dfloat hostVal; // use a stack host variable for single-element copies
    std::stringstream name;

    switch (dir_index)
    {
    case 1: // ux on y-direction
        x_loc = x0;
        z_loc = z0;
        for (y_loc = 0; y_loc < NY; ++y_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             1, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            // optional: validate idx if you have TOTAL_SIZE available
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << (hostVal / F_M_I_SCALE);
        }
        name << "velProfile_ux_dy_x" << x0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 2: // uy on y-direction
        x_loc = x0;
        z_loc = z0;
        for (y_loc = 0; y_loc < NY; ++y_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             2, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << (hostVal / F_M_I_SCALE);
        }
        name << "velProfile_uy_dy_x" << x0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 3: // uz on y-direction
        x_loc = x0;
        z_loc = z0;
        for (y_loc = 0; y_loc < NY; ++y_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             3, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << (hostVal / F_M_I_SCALE);
        }
        name << "velProfile_uz_dy_x" << x0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 4: // ux on x-direction
        y_loc = y0;
        z_loc = z0;
        for (x_loc = 0; x_loc < NX; ++x_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             1, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << (hostVal / F_M_I_SCALE);
        }
        name << "velProfile_ux_dx_y" << y0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 5: // uy on x-direction
        y_loc = y0;
        z_loc = z0;
        for (x_loc = 0; x_loc < NX; ++x_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             2, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << (hostVal / F_M_I_SCALE);
        }
        name << "velProfile_uy_dx_y" << y0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 6: // uz on x-direction
        y_loc = y0;
        z_loc = z0;
        for (x_loc = 0; x_loc < NX; ++x_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             3, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << (hostVal / F_M_I_SCALE);
        }
        name << "velProfile_uz_dx_y" << y0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 7: // ux on z-direction
        y_loc = y0;
        x_loc = x0;
        for (z_loc = 0; z_loc < NZ_TOTAL; ++z_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             1, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << (hostVal / F_M_I_SCALE);
        }
        name << "velProfile_ux_dz_x" << x0 << "_y" << y0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 8: // uy on z-direction
        y_loc = y0;
        x_loc = x0;
        for (z_loc = 0; z_loc < NZ_TOTAL; ++z_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             2, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << (hostVal / F_M_I_SCALE);
        }
        name << "velProfile_uy_dz_x" << x0 << "_y" << y0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 9: // uz on z-direction
        y_loc = y0;
        x_loc = x0;
        for (z_loc = 0; z_loc < NZ_TOTAL; ++z_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             3, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << (hostVal / F_M_I_SCALE);
        }
        name << "velProfile_uz_dz_x" << x0 << "_y" << y0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    default:
        std::cerr << "velocityProfile: unknown dir_index " << dir_index << std::endl;
        break;
    }
}

__host__
void omegaProfile(
    dfloat* fMom,
    int dir_index,
    int x0, int y0, int z0,
    unsigned int step
){
    #ifdef OMEGA_FIELD
    std::ostringstream strDataInfo;
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);
    strDataInfo << "step " << step;

    int x_loc, y_loc, z_loc;
    dfloat hostVal; // use a stack host variable for single-element copies
    std::stringstream name;

    switch (dir_index)
    {
    case 1: // omega on x-direction
        y_loc = y0;
        z_loc = z0;
        for (x_loc = 0; x_loc < NX; ++x_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             M_OMEGA_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << hostVal;
        }
        name << "omegaProfile_dx_y" << y0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 2: // omega on y-direction
        x_loc = x0;
        z_loc = z0;
        for (y_loc = 0; y_loc < NY; ++y_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             M_OMEGA_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            // optional: validate idx if you have TOTAL_SIZE available
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << hostVal;
        }
        name << "omegaProfile_dy_x" << x0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 3: // omega on z-direction
        y_loc = y0;
        x_loc = x0;
        for (z_loc = 0; z_loc < NZ_TOTAL; ++z_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             M_OMEGA_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << hostVal;
        }
        name << "omegaProfile_dz_x" << x0 << "_y" << y0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    default:
        std::cerr << "omegaProfile: unknown dir_index " << dir_index << std::endl;
        break;
    }
    #endif //OMEGA_FIELD
}

__host__
void phiProfile(
    dfloat* fMom,
    int dir_index,
    int x0, int y0, int z0,
    unsigned int step
){
    #ifdef PHI_DIST
    std::ostringstream strDataInfo;
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);
    strDataInfo << "step " << step;

    int x_loc, y_loc, z_loc;
    dfloat hostVal;
    std::stringstream name;

    switch (dir_index)
    {
    case 1: // phi on x-direction
        y_loc = y0;
        z_loc = z0;
        for (x_loc = 0; x_loc < NX; ++x_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             M3_PHI_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << hostVal;
        }
        name << "phiProfile_dx_y" << y0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 2: // phi on y-direction
        x_loc = x0;
        z_loc = z0;
        for (y_loc = 0; y_loc < NY; ++y_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             M3_PHI_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << hostVal;
        }
        name << "phiProfile_dy_x" << x0 << "_z" << z0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    case 3: // phi on z-direction
        y_loc = y0;
        x_loc = x0;
        for (z_loc = 0; z_loc < NZ_TOTAL; ++z_loc) {
            size_t idx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
                             M3_PHI_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
            checkCudaErrors(cudaMemcpy(&hostVal, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo << "\t" << hostVal;
        }
        name << "phiProfile_dz_x" << x0 << "_y" << y0;
        saveTreatData(name.str(), strDataInfo.str(), step);
        break;

    default:
        std::cerr << "phiProfile: unknown dir_index " << dir_index << std::endl;
        break;
    }
    #endif //PHI_DIST
}

#ifdef CONFORMATION_TENSOR
__host__
void conformationProfile(
    dfloat* fMom,
    int dir_index,
    int x0, int y0, int z0,
    unsigned int step
){
    std::ostringstream strAxx;
    std::ostringstream strAxy;
    std::ostringstream strAxz;
    std::ostringstream strAyy;
    std::ostringstream strAyz;
    std::ostringstream strAzz;
    strAxx << std::scientific << std::setprecision(6) << "step " << step;
    strAxy << std::scientific << std::setprecision(6) << "step " << step;
    strAxz << std::scientific << std::setprecision(6) << "step " << step;
    strAyy << std::scientific << std::setprecision(6) << "step " << step;
    strAyz << std::scientific << std::setprecision(6) << "step " << step;
    strAzz << std::scientific << std::setprecision(6) << "step " << step;

    auto appendPoint = [&](int x_loc, int y_loc, int z_loc) {
        dfloat hostAxx;
        dfloat hostAxy;
        dfloat hostAxz;
        dfloat hostAyy;
        dfloat hostAyz;
        dfloat hostAzz;

        const size_t idxAxx = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
            A_XX_C_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
        const size_t idxAxy = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
            A_XY_C_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
        const size_t idxAxz = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
            A_XZ_C_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
        const size_t idxAyy = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
            A_YY_C_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
        const size_t idxAyz = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
            A_YZ_C_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);
        const size_t idxAzz = idxMom(x_loc % BLOCK_NX, y_loc % BLOCK_NY, z_loc % BLOCK_NZ,
            A_ZZ_C_INDEX, x_loc / BLOCK_NX, y_loc / BLOCK_NY, z_loc / BLOCK_NZ);

        checkCudaErrors(cudaMemcpy(&hostAxx, fMom + idxAxx, sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&hostAxy, fMom + idxAxy, sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&hostAxz, fMom + idxAxz, sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&hostAyy, fMom + idxAyy, sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&hostAyz, fMom + idxAyz, sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&hostAzz, fMom + idxAzz, sizeof(dfloat), cudaMemcpyDeviceToHost));

        strAxx << "\t" << (hostAxx - CONF_ZERO);
        strAxy << "\t" << (hostAxy - CONF_ZERO);
        strAxz << "\t" << (hostAxz - CONF_ZERO);
        strAyy << "\t" << (hostAyy - CONF_ZERO);
        strAyz << "\t" << (hostAyz - CONF_ZERO);
        strAzz << "\t" << (hostAzz - CONF_ZERO);
    };

    std::stringstream suffix;
    switch (dir_index)
    {
    case 1:
        for (int y_loc = 0; y_loc < NY; ++y_loc) {
            appendPoint(x0, y_loc, z0);
        }
        suffix << "_dy_x" << x0 << "_z" << z0;
        break;

    case 2:
        for (int x_loc = 0; x_loc < NX; ++x_loc) {
            appendPoint(x_loc, y0, z0);
        }
        suffix << "_dx_y" << y0 << "_z" << z0;
        break;

    case 3:
        for (int z_loc = 0; z_loc < NZ_TOTAL; ++z_loc) {
            appendPoint(x0, y0, z_loc);
        }
        suffix << "_dz_x" << x0 << "_y" << y0;
        break;

    default:
        std::cerr << "conformationProfile: unknown dir_index " << dir_index << std::endl;
        return;
    }

    saveTreatData(std::string("confProfile_Axx") + suffix.str(), strAxx.str(), step);
    saveTreatData(std::string("confProfile_Axy") + suffix.str(), strAxy.str(), step);
    saveTreatData(std::string("confProfile_Axz") + suffix.str(), strAxz.str(), step);
    saveTreatData(std::string("confProfile_Ayy") + suffix.str(), strAyy.str(), step);
    saveTreatData(std::string("confProfile_Ayz") + suffix.str(), strAyz.str(), step);
    saveTreatData(std::string("confProfile_Azz") + suffix.str(), strAzz.str(), step);
}
#endif //CONFORMATION_TENSOR




__host__
void computeNusseltNumber(
    dfloat* h_fMom,
    dfloat* fMom,
    unsigned int step,
    size_t zOffset
){
    //copy full macroscopic field
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_fMom+zOffset, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    #ifdef THERMAL_MODEL

    
    int x0 = 0;
    int x1 = 1;
    int x2 = NX-1;
    int x3 = NX-2;
    dfloat C_x0;
    dfloat C_x1;
    dfloat C_x2;
    dfloat C_x3;
    dfloat Nu_sum = 0.0_df;


    for (int z = 0; z <NZ_TOTAL; z++){
        for(int y = 0; y< NY-0;y++){
            C_x0 = h_fMom[idxMom(x0%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x0/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            C_x1 = h_fMom[idxMom(x1%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x1/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            C_x2 = h_fMom[idxMom(x2%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x2/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            C_x3 = h_fMom[idxMom(x3%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x3/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

            Nu_sum +=-(C_x1 - C_x0);
            Nu_sum +=(C_x3 - C_x2);
        }
    }

    Nu_sum /= (2*(NY-2)*NZ_TOTAL);
    Nu_sum = Nu_sum/(T_DELTA_T/L);

    strDataInfo <<"step,"<< step<< "," << Nu_sum;// << "," << mean_counter;
    saveTreatData("_Nu_mean",strDataInfo.str(),step);

    #endif //THERMAL_MODEL
}


__host__
void computeTurbulentEnergies(
    dfloat* h_fMom,
    dfloat* fMom,
    dfloat* fMom_mean,
    unsigned int step
){

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    //Curent values
    dfloat t_ux0, t_uy0,t_uz0;
    dfloat t_mxx0,t_mxy0,t_mxz0,t_myy0,t_myz0,t_mzz0;


    dfloat Sxx = 0;
    dfloat Sxy = 0;
    dfloat Sxz = 0;
    dfloat Syy = 0;
    dfloat Syz = 0;
    dfloat Szz = 0;
    dfloat SS = 0;
    int count = 0;



    //fluctuation values
    dfloat f_ux, f_uy, f_uz; //NO IDEA WHY IT GIVES A WARNING FOR USED

    dfloat f_Sxx = 0.0_df;
    dfloat f_Sxy = 0.0_df;
    dfloat f_Sxz = 0.0_df;
    dfloat f_Syy = 0.0_df;
    dfloat f_Syz = 0.0_df;
    dfloat f_Szz = 0.0_df;

    dfloat f_SS = 0.0_df;

    //mean values;
    dfloat m_ux = 0.0_df;
    dfloat m_uy = 0.0_df;
    dfloat m_uz = 0.0_df;

    dfloat m_Sxx = 0.0_df;
    dfloat m_Sxy = 0.0_df;
    dfloat m_Sxz = 0.0_df;
    dfloat m_Syy = 0.0_df;
    dfloat m_Syz = 0.0_df;
    dfloat m_Szz = 0.0_df;

#pragma warning(push)
#pragma warning(disable: 4804)
    dfloat mean_counter = 1.0_df/((dfloat)(step/MACR_SAVE)+1.0_df);
    count = 0;
#pragma warning(pop)

    //left side of the equation
    for (int z = 0 ; z <NZ_TOTAL; z++){
        for(int y = 0; y< NY;y++){
            for(int x = 0; x< NX;x++){
                t_ux0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_uy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_uz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                t_mxx0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_mxy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_mxz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_myy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_myz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_mzz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MZZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                Sxx = (as2/(2*TAU))*(t_ux0*t_ux0-t_mxx0);
                Sxy = (as2/(2*TAU))*(t_ux0*t_uy0-t_mxy0);
                Sxz = (as2/(2*TAU))*(t_ux0*t_uz0-t_mxz0);

                Syy = (as2/(2*TAU))*(t_uy0*t_uy0-t_myy0);
                Syz = (as2/(2*TAU))*(t_uy0*t_uz0-t_myz0);

                Szz = (as2/(2*TAU))*(t_uz0*t_uz0-t_mzz0);
                SS += ( Sxx * Sxx + 
                        Syy * Syy + 
                        Szz * Szz + 2*(
                        Sxy * Sxy + 
                        Sxz * Sxz + 
                        Syz * Syz)) ;

                //STORE AND UPDATE MEANS

                //retrive mean values
                m_ux = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_uy = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_uz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                
                m_Sxx = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_Sxy = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_Sxz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_Syy = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_Syz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_Szz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MZZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                //update and store mean values
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_ux + (t_ux0 - m_ux)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uy + (t_uy0 - m_uy)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uz + (t_uz0 - m_uz)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Sxx + (Sxx - m_Sxx)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Sxy + (Sxy - m_Sxy)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Sxz + (Sxz - m_Sxz)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Syy + (Syy - m_Syy)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Syz + (Syz - m_Syz)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MZZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Szz + (Szz - m_Szz)*(mean_counter);
            
                f_ux = t_ux0 - m_ux;
                f_uy = t_uy0 - m_uy;
                f_uz = t_uz0 - m_uz;
                f_Sxx = Sxx - m_Sxx;
                f_Sxy = Sxy - m_Sxy;
                f_Sxz = Sxz - m_Sxz;
                f_Syy = Syy - m_Syy;
                f_Syz = Syz - m_Syz;
                f_Szz = Szz - m_Szz;

                f_SS += ( f_Sxx * f_Sxx + f_Syy * f_Syy + f_Szz * f_Szz + 2*( f_Sxy * f_Sxy + f_Sxz * f_Sxz + f_Syz * f_Syz));                        


                count++;
            }
        }
    }

    SS = SS/(NX*NY*NZ_TOTAL);
    f_SS = f_SS / (count);
    dfloat epsilon = 2.0_df*((TAU-0.5_df)/3.0_df)*f_SS;

    strDataInfo <<"step,"<< step<< "," << SS << "," << epsilon;
    saveTreatData("_turbulentData",strDataInfo.str(),step);
}

__host__
void copyMacroscopic(
    dfloat* h_fMom,
    dfloat* fMom,
    unsigned int step,
    size_t zOffset
){
    //copy full macroscopic field
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_fMom+zOffset, fMom, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    int y_wall;
    dfloat t_ux0, t_uy0,t_uz0;
    dfloat t_rho0;
    dfloat t_Mxx, t_Mxy, t_Mxz, t_Myy, t_Myz, t_Mzz;
    dfloat t_omega0, t_eta0;
    dfloat inv_eta;


    dfloat rho_0[NY];

    dfloat ux_1[NY];
    dfloat uy_1[NY];
    dfloat uz_1[NY];

    dfloat ux_ux[NY];
    dfloat uy_uy[NY];
    dfloat uz_uz[NY];
    dfloat ux_uy[NY];
    dfloat ux_uz[NY];
    dfloat uy_uz[NY];

    dfloat ux_3[NY];
    dfloat uy_3[NY];
    dfloat uz_3[NY];

    dfloat ux_4[NY];
    dfloat uy_4[NY];
    dfloat uz_4[NY];

    dfloat Sxx[NY];
    dfloat Sxy[NY];
    dfloat Sxz[NY];
    dfloat Syy[NY];
    dfloat Syz[NY];
    dfloat Szz[NY];

    dfloat SGxx[NY];
    dfloat SGxy[NY];
    dfloat SGxz[NY];
    dfloat SGyy[NY];
    dfloat SGyz[NY];
    dfloat SGzz[NY];

    dfloat yield_prob[NY];
    dfloat inv_omega[NY];
    dfloat inv_omega_2[NY];

    int count_prob[NY];
    int count_visc[NY];

    for (int y_wall = 0; y_wall <NY;y_wall++){
        rho_0[y_wall] = 0;

        ux_1[y_wall] = 0;
        uy_1[y_wall] = 0;
        uz_1[y_wall] = 0;

        ux_ux[y_wall] = 0;
        uy_uy[y_wall] = 0;
        uz_uz[y_wall] = 0;
        ux_uy[y_wall] = 0;
        ux_uz[y_wall] = 0;
        uy_uz[y_wall] = 0;

        ux_3[y_wall] = 0;
        uy_3[y_wall] = 0;
        uz_3[y_wall] = 0;

        ux_4[y_wall] = 0;
        uy_4[y_wall] = 0;
        uz_4[y_wall] = 0;

        Sxx[y_wall] = 0;
        Sxy[y_wall] = 0;
        Sxz[y_wall] = 0;
        Syy[y_wall] = 0;
        Syz[y_wall] = 0;
        Szz[y_wall] = 0;

        SGxx[y_wall] = 0;
        SGxy[y_wall] = 0;
        SGxz[y_wall] = 0;
        SGyy[y_wall] = 0;
        SGyz[y_wall] = 0;
        SGzz[y_wall] = 0;

        yield_prob[y_wall] = 0;
        inv_omega[y_wall] = 0;
        inv_omega_2[y_wall] = 0;

        count_prob[y_wall]  = 0;
        count_visc[y_wall]  = 0;

    }


    for(int y = 0; y< NY;y++){

            y_wall = y;



        for (int z = 0 ; z <NZ_TOTAL; z++){
            for(int x = 0; x< NX;x++){

                //current lattice value
                t_rho0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_RHO_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] + RHO_0;
                t_ux0 =  h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_I_SCALE;
                t_uy0 =  h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_I_SCALE;
                t_uz0 =  h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_I_SCALE;

                rho_0[y_wall] += t_rho0;

                ux_1[y_wall] += t_ux0;
                uy_1[y_wall] += t_uy0;
                uz_1[y_wall] += t_uz0;

                ux_ux[y_wall] += t_ux0 * t_ux0;
                uy_uy[y_wall] += t_uy0 * t_uy0;
                uz_uz[y_wall] += t_uz0 * t_uz0;
                ux_uy[y_wall] += t_ux0 * t_uy0;
                ux_uz[y_wall] += t_ux0 * t_uz0;
                uy_uz[y_wall] += t_uy0 * t_uz0;

                ux_3[y_wall] += t_ux0*t_ux0*t_ux0;
                uy_3[y_wall] += t_uy0*t_uy0*t_uy0;
                uz_3[y_wall] += t_uz0*t_uz0*t_uz0;

                ux_4[y_wall] += t_ux0*t_ux0*t_ux0*t_ux0;
                uy_4[y_wall] += t_uy0*t_uy0*t_uy0*t_uy0;
                uz_4[y_wall] += t_uz0*t_uz0*t_uz0*t_uz0;

                t_Mxx = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_II_SCALE;
                t_Mxy = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_IJ_SCALE;
                t_Mxz = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_IJ_SCALE;
                t_Myy = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_II_SCALE;
                t_Myz = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_IJ_SCALE;
                t_Mzz = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MZZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_II_SCALE;

                #ifdef OMEGA_FIELD
                    t_omega0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_OMEGA_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                #else
                    t_omega0 = OMEGA;
                #endif //OMEGA_FIELD
                //correction for pre-collsion values  
                t_Mxx = (t_Mxx - t_omega0 * t_ux0 * t_ux0 - (1.0 - t_omega0/2)*0)/(1.0 - t_omega0);
                t_Mxy = (t_Mxy - t_omega0 * t_ux0 * t_uy0 - (1.0 - t_omega0/2)*0)/(1.0 - t_omega0);
                t_Mxz = (t_Mxz - t_omega0 * t_ux0 * t_uz0 - (1.0 - t_omega0/2)*(t_ux0*FZ))/(1.0 - t_omega0);
                t_Myy = (t_Myy - t_omega0 * t_uy0 * t_uy0 - (1.0 - t_omega0/2)*0)/(1.0 - t_omega0);
                t_Myz = (t_Myz - t_omega0 * t_uy0 * t_uz0 - (1.0 - t_omega0/2)*(t_uy0*FZ))/(1.0 - t_omega0);
                t_Mzz = (t_Mzz - t_omega0 * t_uz0 * t_uz0 - (1.0 - t_omega0/2)*(t_uz0*FZ*2))/(1.0 - t_omega0);

                //computing stress
                t_Mxx = t_Mxx - t_ux0 * t_ux0 + (FX * t_ux0 + t_ux0 * FX)/2;
                t_Mxy = t_Mxy - t_ux0 * t_uy0 + (FX * t_uy0 + t_ux0 * FY)/2;
                t_Mxz = t_Mxz - t_ux0 * t_uz0 + (FX * t_uz0 + t_ux0 * FZ)/2;
                t_Myy = t_Myy - t_uy0 * t_uy0 + (FY * t_uy0 + t_uy0 * FY)/2;
                t_Myz = t_Myz - t_uy0 * t_uz0 + (FY * t_uz0 + t_uy0 * FZ)/2;
                t_Mzz = t_Mzz - t_uz0 * t_uz0 + (FZ * t_uz0 + t_uz0 * FZ)/2;

                t_Mxx = (1.0-t_omega0/2)*t_Mxx;
                t_Mxy = (1.0-t_omega0/2)*t_Mxy;
                t_Mxz = (1.0-t_omega0/2)*t_Mxz;
                t_Myy = (1.0-t_omega0/2)*t_Myy;
                t_Myz = (1.0-t_omega0/2)*t_Myz;
                t_Mzz = (1.0-t_omega0/2)*t_Mzz;

                Sxx[y_wall] += t_Mxx;
                Sxy[y_wall] += t_Mxy;
                Sxz[y_wall] += t_Mxz;
                Syy[y_wall] += t_Myy;
                Syz[y_wall] += t_Myz;
                Szz[y_wall] += t_Mzz;

                //computing stress times gamma
                t_Mxx *= t_Mxx;
                t_Mxy *= t_Mxy;
                t_Mxz *= t_Mxz;
                t_Myy *= t_Myy;
                t_Myz *= t_Myz;
                t_Mzz *= t_Mzz;

                inv_eta = 6*t_omega0/(2-t_omega0);

                SGxx[y_wall] += t_Mxx * inv_eta;
                SGxy[y_wall] += t_Mxy * inv_eta;
                SGxz[y_wall] += t_Mxz * inv_eta;
                SGyy[y_wall] += t_Myy * inv_eta;
                SGyz[y_wall] += t_Myz * inv_eta;
                SGzz[y_wall] += t_Mzz * inv_eta;


                if(t_omega0 < OMEGA_CUTOFF){
                    yield_prob[y_wall] += 1.0;
                    count_prob[y_wall] = count_prob[y_wall] + 1;
                }else{
                    inv_omega[y_wall] += 1.0/t_omega0;
                    inv_omega_2[y_wall] += 1.0/(t_omega0*t_omega0);
                    count_visc[y_wall] = count_visc[y_wall] + 1;
                }

            }
        }
    }

    for (int y_wall = 0; y_wall <NY;y_wall++){
        rho_0[y_wall] /= (NX*NZ_TOTAL);

        ux_1[y_wall] /= (NX*NZ_TOTAL);
        uy_1[y_wall] /= (NX*NZ_TOTAL);
        uz_1[y_wall] /= (NX*NZ_TOTAL);

        ux_ux[y_wall] /= (NX*NZ_TOTAL);
        uy_uy[y_wall] /= (NX*NZ_TOTAL);
        uz_uz[y_wall] /= (NX*NZ_TOTAL);
        ux_uy[y_wall] /= (NX*NZ_TOTAL);
        ux_uz[y_wall] /= (NX*NZ_TOTAL);
        uy_uz[y_wall] /= (NX*NZ_TOTAL);

        ux_3[y_wall] /= (NX*NZ_TOTAL);
        uy_3[y_wall] /= (NX*NZ_TOTAL);
        uz_3[y_wall] /= (NX*NZ_TOTAL);

        ux_4[y_wall] /= (NX*NZ_TOTAL);
        uy_4[y_wall] /= (NX*NZ_TOTAL);
        uz_4[y_wall] /= (NX*NZ_TOTAL);

        Sxx[y_wall] /= (NX*NZ_TOTAL);
        Sxy[y_wall] /= (NX*NZ_TOTAL);
        Sxz[y_wall] /= (NX*NZ_TOTAL);
        Syy[y_wall] /= (NX*NZ_TOTAL);
        Syz[y_wall] /= (NX*NZ_TOTAL);
        Szz[y_wall] /= (NX*NZ_TOTAL);

        SGxx[y_wall] /= (NX*NZ_TOTAL);
        SGxy[y_wall] /= (NX*NZ_TOTAL);
        SGxz[y_wall] /= (NX*NZ_TOTAL);
        SGyy[y_wall] /= (NX*NZ_TOTAL);
        SGyz[y_wall] /= (NX*NZ_TOTAL);
        SGzz[y_wall] /= (NX*NZ_TOTAL);

        yield_prob[y_wall] /= (NX*NZ_TOTAL);
        inv_omega[y_wall] /= count_visc[y_wall];
        inv_omega_2[y_wall] /= count_visc[y_wall];

    }

    std::ostringstream strDataInfo_rho_0("");

    std::ostringstream strDataInfo_ux_1("");
    std::ostringstream strDataInfo_uy_1("");
    std::ostringstream strDataInfo_uz_1("");

    std::ostringstream strDataInfo_ux_ux("");
    std::ostringstream strDataInfo_uy_uy("");
    std::ostringstream strDataInfo_uz_uz("");
    std::ostringstream strDataInfo_ux_uy("");
    std::ostringstream strDataInfo_ux_uz("");
    std::ostringstream strDataInfo_uy_uz("");

    std::ostringstream strDataInfo_ux_3("");
    std::ostringstream strDataInfo_uy_3("");
    std::ostringstream strDataInfo_uz_3("");

    std::ostringstream strDataInfo_ux_4("");
    std::ostringstream strDataInfo_uy_4("");
    std::ostringstream strDataInfo_uz_4("");

    std::ostringstream strDataInfo_Sxx("");
    std::ostringstream strDataInfo_Sxy("");
    std::ostringstream strDataInfo_Sxz("");
    std::ostringstream strDataInfo_Syy("");
    std::ostringstream strDataInfo_Syz("");
    std::ostringstream strDataInfo_Szz("");

    std::ostringstream strDataInfo_SGxx("");
    std::ostringstream strDataInfo_SGxy("");
    std::ostringstream strDataInfo_SGxz("");
    std::ostringstream strDataInfo_SGyy("");
    std::ostringstream strDataInfo_SGyz("");
    std::ostringstream strDataInfo_SGzz("");

    std::ostringstream strDataInfo_yield_prob("");
    std::ostringstream strDataInfo_inv_omega("");
    std::ostringstream strDataInfo_inv_omega_2("");

    strDataInfo_rho_0 <<"step,"<< step;

    strDataInfo_ux_1 <<"step,"<< step;
    strDataInfo_uy_1 <<"step,"<< step;
    strDataInfo_uz_1 <<"step,"<< step;

    strDataInfo_ux_ux <<"step,"<< step;
    strDataInfo_uy_uy <<"step,"<< step;
    strDataInfo_uz_uz <<"step,"<< step;
    strDataInfo_ux_uy <<"step,"<< step;
    strDataInfo_ux_uz <<"step,"<< step;
    strDataInfo_uy_uz <<"step,"<< step;

    strDataInfo_ux_3 <<"step,"<< step;
    strDataInfo_uy_3 <<"step,"<< step;
    strDataInfo_uz_3 <<"step,"<< step;

    strDataInfo_ux_4 <<"step,"<< step;
    strDataInfo_uy_4 <<"step,"<< step;
    strDataInfo_uz_4 <<"step,"<< step;

    strDataInfo_Sxx <<"step,"<< step;
    strDataInfo_Sxy <<"step,"<< step;
    strDataInfo_Sxz <<"step,"<< step;
    strDataInfo_Syy <<"step,"<< step;
    strDataInfo_Syz <<"step,"<< step;
    strDataInfo_Szz <<"step,"<< step;

    strDataInfo_SGxx <<"step,"<< step;
    strDataInfo_SGxy <<"step,"<< step;
    strDataInfo_SGxz <<"step,"<< step;
    strDataInfo_SGyy <<"step,"<< step;
    strDataInfo_SGyz <<"step,"<< step;
    strDataInfo_SGzz <<"step,"<< step;

    strDataInfo_yield_prob <<"step,"<< step;
    strDataInfo_inv_omega <<"step,"<< step;
    strDataInfo_inv_omega_2 <<"step,"<< step;

    for(int y = 0; y< NY;y++){

        strDataInfo_rho_0 << "," << rho_0[y];

        strDataInfo_ux_1 << "," << ux_1[y];
        strDataInfo_uy_1 << "," << uy_1[y];
        strDataInfo_uz_1 << "," << uz_1[y];

        strDataInfo_ux_ux  << "," << ux_ux[y];
        strDataInfo_uy_uy  << "," << uy_uy[y];
        strDataInfo_uz_uz  << "," << uz_uz[y];
        strDataInfo_ux_uy  << "," << ux_uy[y];
        strDataInfo_ux_uz  << "," << ux_uz[y];
        strDataInfo_uy_uz  << "," << uy_uz[y];

        strDataInfo_ux_3  << "," << ux_3[y];
        strDataInfo_uy_3  << "," << uy_3[y];
        strDataInfo_uz_3  << "," << uz_3[y];

        strDataInfo_ux_4  << "," << ux_4[y];
        strDataInfo_uy_4  << "," << uy_4[y];
        strDataInfo_uz_4  << "," << uz_4[y];

        strDataInfo_Sxx  << "," << Sxx[y];
        strDataInfo_Sxy  << "," << Sxy[y];
        strDataInfo_Sxz  << "," << Sxz[y];
        strDataInfo_Syy  << "," << Syy[y];
        strDataInfo_Syz  << "," << Syz[y];
        strDataInfo_Szz  << "," << Szz[y];

        strDataInfo_SGxx  << "," << SGxx[y];
        strDataInfo_SGxy  << "," << SGxy[y];
        strDataInfo_SGxz  << "," << SGxz[y];
        strDataInfo_SGyy  << "," << SGyy[y];
        strDataInfo_SGyz  << "," << SGyz[y];
        strDataInfo_SGzz  << "," << SGzz[y];

        strDataInfo_yield_prob << "," << yield_prob[y];
        strDataInfo_inv_omega  << "," << inv_omega[y];
        strDataInfo_inv_omega_2  << "," << inv_omega_2[y];
    }

    saveTreatData("_turbulent_rho",strDataInfo_rho_0.str(),step);

    saveTreatData("_turbulent_ux1",strDataInfo_ux_1.str(),step);
    saveTreatData("_turbulent_uy1",strDataInfo_uy_1.str(),step);
    saveTreatData("_turbulent_uz1",strDataInfo_uz_1.str(),step);

    saveTreatData("_turbulent_ux_ux",strDataInfo_ux_ux.str(),step);
    saveTreatData("_turbulent_uy_uy",strDataInfo_uy_uy.str(),step);
    saveTreatData("_turbulent_uz_uz",strDataInfo_uz_uz.str(),step);
    saveTreatData("_turbulent_ux_uy",strDataInfo_ux_uy.str(),step);
    saveTreatData("_turbulent_ux_uz",strDataInfo_ux_uz.str(),step);
    saveTreatData("_turbulent_uy_uz",strDataInfo_uy_uz.str(),step);

    saveTreatData("_turbulent_ux3",strDataInfo_ux_3.str(),step);
    saveTreatData("_turbulent_uy3",strDataInfo_uy_3.str(),step);
    saveTreatData("_turbulent_uz3",strDataInfo_uz_3.str(),step);

    saveTreatData("_turbulent_ux4",strDataInfo_ux_4.str(),step);
    saveTreatData("_turbulent_uy4",strDataInfo_uy_4.str(),step);
    saveTreatData("_turbulent_uz4",strDataInfo_uz_4.str(),step);

    saveTreatData("_turbulent_Sxx",strDataInfo_Sxx.str(),step);
    saveTreatData("_turbulent_Syy",strDataInfo_Syy.str(),step);
    saveTreatData("_turbulent_Szz",strDataInfo_Szz.str(),step);
    saveTreatData("_turbulent_Sxy",strDataInfo_Sxy.str(),step);
    saveTreatData("_turbulent_Sxz",strDataInfo_Sxz.str(),step);
    saveTreatData("_turbulent_Syz",strDataInfo_Syz.str(),step);


    saveTreatData("_turbulent_SGxx",strDataInfo_SGxx.str(),step);
    saveTreatData("_turbulent_SGyy",strDataInfo_SGyy.str(),step);
    saveTreatData("_turbulent_SGzz",strDataInfo_SGzz.str(),step);
    saveTreatData("_turbulent_SGxy",strDataInfo_SGxy.str(),step);
    saveTreatData("_turbulent_SGxz",strDataInfo_SGxz.str(),step);
    saveTreatData("_turbulent_SGyz",strDataInfo_SGyz.str(),step);

    saveTreatData("_turbulent_yield_prob",strDataInfo_yield_prob.str(),step);
    saveTreatData("_turbulent_inv_omega",strDataInfo_inv_omega.str(),step);
    saveTreatData("_turbulent_inv_omega_2",strDataInfo_inv_omega_2.str(),step);


    // FOR CORRELATIONS

    int x0, y0, z0;

    x0 = NX/2;
    z0 = NZ_TOTAL/2;
    //x direction
    velocityProfile(fMom,4,x0,(2*del)/96,z0,step); //ux
    velocityProfile(fMom,5,x0,(2*del)/96,z0,step); //uy
    velocityProfile(fMom,6,x0,(2*del)/96,z0,step); //uz

    velocityProfile(fMom,4,x0,(6*del)/96,z0,step); //ux
    velocityProfile(fMom,5,x0,(6*del)/96,z0,step); //uy
    velocityProfile(fMom,6,x0,(6*del)/96,z0,step); //uz

    velocityProfile(fMom,4,x0,(52*del)/96,z0,step); //ux
    velocityProfile(fMom,5,x0,(52*del)/96,z0,step); //uy
    velocityProfile(fMom,6,x0,(52*del)/96,z0,step); //uz

    velocityProfile(fMom,4,x0,(96*del)/96,z0,step); //ux
    velocityProfile(fMom,5,x0,(96*del)/96,z0,step); //uy
    velocityProfile(fMom,6,x0,(96*del)/96,z0,step); //uz

    //Z DIRECITON
    velocityProfile(fMom,7,x0,(2*del)/96,z0,step); //ux
    velocityProfile(fMom,8,x0,(2*del)/96,z0,step); //uy
    velocityProfile(fMom,9,x0,(2*del)/96,z0,step); //uz

    velocityProfile(fMom,7,x0,(6*del)/96,z0,step); //ux
    velocityProfile(fMom,8,x0,(6*del)/96,z0,step); //uy
    velocityProfile(fMom,9,x0,(6*del)/96,z0,step); //uz

    velocityProfile(fMom,7,x0,(52*del)/96,z0,step); //ux
    velocityProfile(fMom,8,x0,(52*del)/96,z0,step); //uy
    velocityProfile(fMom,9,x0,(52*del)/96,z0,step); //uz

    velocityProfile(fMom,7,x0,(96*del)/96,z0,step); //ux
    velocityProfile(fMom,8,x0,(96*del)/96,z0,step); //uy
    velocityProfile(fMom,9,x0,(96*del)/96,z0,step); //uz


    rhoProfile(fMom,3,x0,(2*del)/96,z0,step);  //rho
    rhoProfile(fMom,3,x0,(6*del)/96,z0,step);  //rho
    rhoProfile(fMom,3,x0,(52*del)/96,z0,step); //rho
    rhoProfile(fMom,3,x0,(96*del)/96,z0,step); //rho

    omegaProfile(fMom,3,x0,(2*del)/96,z0,step);  //omega
    omegaProfile(fMom,3,x0,(6*del)/96,z0,step);  //omega
    omegaProfile(fMom,3,x0,(52*del)/96,z0,step); //omega
    omegaProfile(fMom,3,x0,(96*del)/96,z0,step); //omega

    /*

    // strain tensor

    dfloat Sxx[NY/2];
    dfloat Sxy[NY/2];
    dfloat Sxz[NY/2];
    dfloat Syy[NY/2];
    dfloat Syz[NY/2];
    dfloat Szz[NY/2];


    //kinetic energy and invariants
    dfloat turb_kinetic[NY/2];
    dfloat invariant2[NY/2];
    dfloat invariant3[NY/2];
    dfloat R[3][3];
    dfloat aij[3][3];



    //Update the mean field values
    dfloat mean_counter = 1.0/((dfloat)(step/MACR_SAVE)+1.0);




    // mean velocity values

    ux_mean[y_wall] /= (2*NX*NZ_TOTAL);;
    uy_mean[y_wall] /= (2*NX*NZ_TOTAL);;
    uz_mean[y_wall] /= (2*NX*NZ_TOTAL);;


    for(int y = 0; y< NY;y++){
        //distance to any wall
        y_wall = (y <= (NY-1)/2) ? y : (NY-1 - y);
        

        for (int z = 0 ; z <NZ_TOTAL; z++){
            for(int x = 0; x< NX;x++){
                //current lattice value
            }
        }

        // Build Reynolds stress tensor R_ij
        R[0][0] = ux_fluct[y_wall];      // <u'x u'x>
        R[1][1] = uy_fluct[y_wall];      // <u'y u'y>
        R[2][2] = uz_fluct[y_wall];      // <u'z u'z>
        R[0][1] = R[1][0] = uxuy_fluct[y_wall];
        R[0][2] = R[2][0] = uxuz_fluct[y_wall];
        R[1][2] = R[2][1] = uyuz_fluct[y_wall];

        // Turbulent kinetic energy
        turb_kinetic[y_wall] = 0.5 * (R[0][0] + R[1][1] + R[2][2]);

        // Anisotropy tensor a_ij = R_ij/(2k) - delta_ij/3
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                aij[i][j] = R[i][j] / (2.0 * turb_kinetic[y_wall]);
                if (i == j) aij[i][j] -= 1.0/3.0;
            }
        }

        // Invariant II = -0.5 * a_ij a_ji
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                invariant2[y_wall] += aij[i][j] * aij[j][i];
            }
        }
        invariant2[y_wall] *= -0.5;

        invariant3[y_wall] = aij[0][0]*(aij[1][1]*aij[2][2] - aij[1][2]*aij[2][1]) -
                            aij[0][1]*(aij[1][0]*aij[2][2] - aij[1][2]*aij[2][0]) +
                            aij[0][2]*(aij[1][0]*aij[2][1] - aij[1][1]*aij[2][0]);


    }

    std::ostringstream strDataInfo_k("");
    std::ostringstream strDataInfo_II("");
    std::ostringstream strDataInfo_III("");

    strDataInfo_k  << "step," << step;
    strDataInfo_II << "step," << step;
    strDataInfo_III<< "step," << step;



    for(int y = 0; y< NY/2;y++){
        strDataInfo_k  << "," << turb_kinetic[y];
        strDataInfo_II << "," << invariant2[y];
        strDataInfo_III<< "," << invariant3[y];
    }

    saveTreatData("_turbulent_k",   strDataInfo_k.str(), step);
    saveTreatData("_turbulent_II",  strDataInfo_II.str(), step);
    saveTreatData("_turbulent_III", strDataInfo_III.str(), step);



    */
}