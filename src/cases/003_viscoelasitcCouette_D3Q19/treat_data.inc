/*
checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom,
    sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS,
    cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());

// --- accumulate global norm difference
dfloat normSum = 0.0;
dfloat normMax = 0.0;
size_t count = 0;

for (int z = 0; z < NZ; ++z) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {

            // --- numerical values (subtract CONF_ZERO baseline)
            dfloat Axx_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ,
                                           A_XX_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            dfloat Axy_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ,
                                           A_XY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            dfloat Ayy_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ,
                                           A_YY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;

            // optional if using full 3D tensor:
            // dfloat Axz_num = h_fMom[idxMom(..., A_XZ_C_INDEX, ...)] - CONF_ZERO;
            // dfloat Ayz_num = h_fMom[idxMom(..., A_YZ_C_INDEX, ...)] - CONF_ZERO;
            // dfloat Azz_num = h_fMom[idxMom(..., A_ZZ_C_INDEX, ...)] - CONF_ZERO;

            // --- reference solution (analytical / unperturbed)
            dfloat Axx_ref = 1.0 + 2.0*Weissenberg_number*Weissenberg_number;
            dfloat Axy_ref = Weissenberg_number;
            dfloat Ayy_ref = 1.0;

            // full tensor refs if needed:
            // dfloat Axz_ref = 0.0;
            // dfloat Ayz_ref = 0.0;
            // dfloat Azz_ref = 1.0;

            // --- differences
            dfloat dAxx = Axx_num - Axx_ref;
            dfloat dAxy = Axy_num - Axy_ref;
            dfloat dAyy = Ayy_num - Ayy_ref;

            // full tensor differences if needed:
            // dfloat dAxz = Axz_num - Axz_ref;
            // dfloat dAyz = Ayz_num - Ayz_ref;
            // dfloat dAzz = Azz_num - Azz_ref;

            // --- Frobenius norm contribution (2D symmetric tensor)
            dfloat locNorm2 = dAxx*dAxx + dAyy*dAyy + 2.0*dAxy*dAxy;

            // for full 3D symmetric tensor:
            // dfloat locNorm2 = dAxx*dAxx + dAyy*dAyy + dAzz*dAzz
            //                 + 2.0*(dAxy*dAxy + dAxz*dAxz + dAyz*dAyz);

            normSum += locNorm2;
            normMax = std::max(normMax, sqrt(locNorm2));
            count++;
        }
    }
}

// --- global Frobenius norm
dfloat frobNorm = sqrt(normSum);          // absolute norm
dfloat frobNormAvg = sqrt(normSum/count); // RMS relative to domain size

// --- save output
if(step == REPORT_SAVE) {
    std::ostringstream strDataHeader;
    strDataHeader << "step,norm,normRMS,normMax";
    saveTreatDataHeader("_data_norm", strDataHeader.str());
}

std::ostringstream strDataInfo;
strDataInfo << std::scientific << std::setprecision(6);
strDataInfo << step*inv_lambda
            << "," << frobNorm
            << "," << frobNormAvg
            << "," << normMax;
saveTreatData("_data_norm", strDataInfo.str(), step,true);


*/


std::vector<dfloat> Aref_vec(NUMBER_LBM_NODES * 3);
std::ifstream fin("A_ref.bin", std::ios::binary);
fin.read(reinterpret_cast<char*>(Aref_vec.data()), Aref_vec.size() * sizeof(dfloat));
fin.close();



checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());


// --- accumulate global norm difference
dfloat normSum = 0.0;
dfloat normMax = 0.0;
size_t count = 0;
size_t idx = 0;

for (int z = 0; z < NZ; ++z) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            // numerical tensor
            dfloat Axx_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XX_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            dfloat Axy_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            dfloat Ayy_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_YY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;

            // reference tensor from numerical steady state
            dfloat Axx_ref = Aref_vec[idx + 0];
            dfloat Axy_ref = Aref_vec[idx + 1];
            dfloat Ayy_ref = Aref_vec[idx + 2];

            // differences
            dfloat dAxx = Axx_num - Axx_ref;
            dfloat dAxy = Axy_num - Axy_ref;
            dfloat dAyy = Ayy_num - Ayy_ref;

            // Frobenius norm contribution (2D symmetric)
            dfloat locNorm2 = dAxx*dAxx + dAyy*dAyy + 2.0*dAxy*dAxy;

            normSum += locNorm2;
            normMax = std::max(normMax, sqrt(locNorm2));
            count++;

            idx += 3;
        }
    }
}

// --- global Frobenius norm
dfloat frobNorm = sqrt(normSum);          // absolute norm
dfloat frobNormAvg = sqrt(normSum/count); // RMS relative to domain size



if(step == REPORT_SAVE) {
    std::ostringstream strDataHeader;
    strDataHeader << "step,norm,normRMS,normMax";
    saveTreatDataHeader("_data_norm", strDataHeader.str());
}

std::ostringstream strDataInfo;
strDataInfo << std::scientific << std::setprecision(6);
strDataInfo << step*inv_lambda
            << "," << frobNorm
            << "," << frobNormAvg
            << "," << normMax;
saveTreatData("_data_norm", strDataInfo.str(), step,true);




//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/*




checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom,
    sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS,
    cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());

// Save numerical steady-state as reference
std::vector<dfloat> Aref_vec(NUMBER_LBM_NODES * 3); // N_COMP = number of tensor components
size_t idx = 0;

for (int z = 0; z < NZ; ++z){
    for (int y = 0; y < NY; ++y){
        for (int x = 0; x < NX; ++x) {
            Aref_vec[idx++] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XX_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            Aref_vec[idx++] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            Aref_vec[idx++] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_YY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            // add other components if using full 3D tensor
        }
    }
}

// save to binary file
std::ofstream fout("A_ref.bin", std::ios::binary);
fout.write(reinterpret_cast<const char*>(Aref_vec.data()), Aref_vec.size()*sizeof(dfloat));
fout.close();

*/