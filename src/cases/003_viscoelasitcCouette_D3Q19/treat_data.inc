// --- analytic UCM/Oldroyd-B steady state (exact fixed point, no file needed)
dfloat Axx_ref = 1.0 + 2.0*Weissenberg_number*Weissenberg_number;
dfloat Axy_ref = Weissenberg_number;
dfloat Ayy_ref = 1.0;

// --- per-component accumulators
dfloat sumSq_xx = 0.0, sumSq_xy = 0.0, sumSq_yy = 0.0;
dfloat max_xx   = 0.0, max_xy   = 0.0, max_yy   = 0.0;
dfloat sum_xx   = 0.0, sum_xy   = 0.0, sum_yy   = 0.0;  // raw field values, not error
size_t count = 0;

for (int z = 0; z < NZ; ++z) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            dfloat Axx_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XX_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            dfloat Axy_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            dfloat Ayy_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_YY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;

            dfloat dAxx = Axx_num - Axx_ref;
            dfloat dAxy = Axy_num - Axy_ref;
            dfloat dAyy = Ayy_num - Ayy_ref;

            sumSq_xx += dAxx*dAxx;
            sumSq_xy += dAxy*dAxy;
            sumSq_yy += dAyy*dAyy;

            max_xx = std::max(max_xx, std::abs(dAxx));
            max_xy = std::max(max_xy, std::abs(dAxy));
            max_yy = std::max(max_yy, std::abs(dAyy));

            sum_xx += Axx_num;
            sum_xy += Axy_num;
            sum_yy += Ayy_num;

            count++;
        }
    }
}

dfloat rms_xx = sqrt(sumSq_xx/count);
dfloat rms_xy = sqrt(sumSq_xy/count);
dfloat rms_yy = sqrt(sumSq_yy/count);

dfloat mean_xx = sum_xx/count;
dfloat mean_xy = sum_xy/count;
dfloat mean_yy = sum_yy/count;

if (step == REPORT_SAVE) {
    std::ostringstream strDataHeader;
    strDataHeader << "step,rms_xx,rms_xy,rms_yy,max_xx,max_xy,max_yy,mean_xx,mean_xy,mean_yy";
    saveTreatDataHeader("_data_norm", strDataHeader.str());
}

std::ostringstream strDataInfo;
strDataInfo << std::scientific << std::setprecision(6);
strDataInfo << step*inv_lambda
            << "," << rms_xx << "," << rms_xy << "," << rms_yy
            << "," << max_xx << "," << max_xy << "," << max_yy
            << "," << mean_xx << "," << mean_xy << "," << mean_yy;
saveTreatData("_data_norm", strDataInfo.str(), step, true);

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
std::ostringstream refDir;
refDir << "data/ref_Wi_" << std::fixed << std::setprecision(3) << Weissenberg_number
       << "_pert_" << std::fixed << std::setprecision(3) << pertubation;
std::string refPath = refDir.str() + "/A_ref.bin";

std::vector<dfloat> Aref_vec(NUMBER_LBM_NODES * 3);
std::ifstream fin(refPath, std::ios::binary);
if (!fin) {
    std::cerr << "ERROR: reference not found at " << refPath
              << " -- run save mode for Wi=" << Weissenberg_number
              << ", pert=" << pertubation << " first.\n";
    std::exit(EXIT_FAILURE);
}
fin.read(reinterpret_cast<char*>(Aref_vec.data()), Aref_vec.size() * sizeof(dfloat));
fin.close();

checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());

dfloat normSum = 0.0;
dfloat normMax = 0.0;
size_t count = 0;
size_t idx = 0;
for (int z = 0; z < NZ; ++z) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            dfloat Axx_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XX_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            dfloat Axy_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            dfloat Ayy_num = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_YY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;

            dfloat Axx_ref = Aref_vec[idx + 0];
            dfloat Axy_ref = Aref_vec[idx + 1];
            dfloat Ayy_ref = Aref_vec[idx + 2];

            dfloat dAxx = Axx_num - Axx_ref;
            dfloat dAxy = Axy_num - Axy_ref;
            dfloat dAyy = Ayy_num - Ayy_ref;

            dfloat locNorm2 = dAxx*dAxx + dAyy*dAyy + 2.0*dAxy*dAxy;
            normSum += locNorm2;
            normMax = std::max(normMax, sqrt(locNorm2));
            count++;
            idx += 3;
        }
    }
}

dfloat frobNorm = sqrt(normSum);
dfloat frobNormAvg = sqrt(normSum/count);

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
saveTreatData("_data_norm", strDataInfo.str(), step, true);


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/*

checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom,
    sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS,
    cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());

std::vector<dfloat> Aref_vec(NUMBER_LBM_NODES * 3);
size_t idx = 0;
for (int z = 0; z < NZ; ++z){
    for (int y = 0; y < NY; ++y){
        for (int x = 0; x < NX; ++x) {
            Aref_vec[idx++] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XX_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            Aref_vec[idx++] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
            Aref_vec[idx++] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_YY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
        }
    }
}

std::ostringstream refDir;
refDir << "data/ref_Wi_" << std::fixed << std::setprecision(3) << Weissenberg_number
       << "_pert_" << std::fixed << std::setprecision(3) << pertubation;
std::error_code ec;
std::filesystem::create_directories(refDir.str(), ec);
if (ec) {
    std::cerr << "ERROR: failed to create directory " << refDir.str()
              << " (" << ec.message() << ")\n";
    return;
}
std::string refPath = refDir.str() + "/A_ref.bin";

std::ofstream fout(refPath, std::ios::binary);
if (!fout) {
    std::cerr << "ERROR: failed to open " << refPath << " for writing\n";
    return;
}
fout.write(reinterpret_cast<const char*>(Aref_vec.data()), Aref_vec.size()*sizeof(dfloat));
if (!fout) {
    std::cerr << "ERROR: write failed for " << refPath << "\n";
}
fout.close();
std::cout << "Saved reference to " << refPath << " ("
          << Aref_vec.size()*sizeof(dfloat) << " bytes)\n";
*/