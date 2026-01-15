#include "particlesReport.cuh"

#ifdef PARTICLE_MODEL

std::string getStrDfloat3(dfloat3 val, std::string sep){
    std::ostringstream strValues("");
    strValues << std::scientific;
    strValues << val.x << sep << val.y << sep << val.z;
    return strValues.str();
}

std::string getStrDfloat4(dfloat4 val, std::string sep){
    std::ostringstream strValues("");
    strValues << std::scientific;
    strValues << val.x << sep << val.y << sep << val.z << sep << val.w;
    return strValues.str();
}


std::string getStrDfloat6(dfloat6 val, std::string sep){
    std::ostringstream strValues("");
    strValues << std::scientific;
    strValues << val.xx << sep << val.yy << sep << val.zz<< sep << val.xy << sep << val.xz << sep << val.yz;
    return strValues.str();
}

void saveParticlesInfo(ParticlesSoA *particles, unsigned int step, std::atomic<bool>& savingMacrParticle){
   
    savingMacrParticle = true;
    std::thread saveThread([particles, step, &savingMacrParticle]() {
        // Names of file to save particle info
        std::string strFilePCenters = getVarFilename("pCenters", step, ".csv");

        // File to save particle info
        std::ofstream outFilePCenter(strFilePCenters.c_str());

        // String with all values as csv
        std::ostringstream strValuesParticles("");
        strValuesParticles << std::scientific;

        // csv separator
        std::string sep = ",";
        // Column names to use in csv
        std::string strColumnNames = "p_number" + sep + "step" + sep;
        strColumnNames += "pos_x" + sep  + "pos_y" + sep  + "pos_z" + sep;
        strColumnNames += "vel_x" + sep  + "vel_y" + sep  + "vel_z" + sep;
        strColumnNames += "w_x" + sep  + "w_y" + sep  + "w_z" + sep;
        strColumnNames += "f_x" + sep  + "f_y" + sep  + "f_z" + sep;
        strColumnNames += "M_x" + sep  + "M_y" + sep  + "M_z" + sep;
        strColumnNames += "I_xx" + sep  + "I_yy" + sep  + "I_zz" + sep + "I_xy" + sep  + "I_xz" + sep  + "I_yz" + sep;
        strColumnNames += "S" + sep;
        strColumnNames += "radius" + sep;
        strColumnNames += "volume" + sep;
        strColumnNames += "movable" + sep;
        strColumnNames += "semi1x" + sep  + "semi1y" + sep  + "semi1z" + sep;
        strColumnNames += "semi2x" + sep  + "semi2y" + sep  + "semi2z" + sep;
        strColumnNames += "semi3x" + sep  + "semi3y" + sep  + "semi3z\n";

        for(int p = 0; p < NUM_PARTICLES; p++){
            ParticleCenter pc = particles->getPCenterArray()[p];

            dfloat3 pos = pc.getPos();
            dfloat3 pos_old = pc.getPos_old();
            dfloat3 vel = pc.getVel();
            dfloat3 vel_old = pc.getVel_old();

            strValuesParticles << p << sep;
            strValuesParticles << step << sep;
            strValuesParticles << getStrDfloat3(pos, sep) << sep;
            strValuesParticles << getStrDfloat3(vel, sep) << sep;
            strValuesParticles << getStrDfloat3(pc.getW(), sep) << sep;
            strValuesParticles << getStrDfloat3(pc.getF(), sep) << sep;
            strValuesParticles << getStrDfloat3(pc.getM(), sep) << sep;
            strValuesParticles << getStrDfloat6(pc.getI(), sep) << sep;
            strValuesParticles << pc.getS() << sep;
            strValuesParticles << pc.getRadius() << sep;
            strValuesParticles << pc.getVolume() << sep;
            strValuesParticles << pc.getMovable() << sep;
            strValuesParticles << getStrDfloat3(pc.getSemiAxis1(), sep) << sep;
            strValuesParticles << getStrDfloat3(pc.getSemiAxis2(), sep) << sep;
            strValuesParticles << getStrDfloat3(pc.getSemiAxis3(), sep) << "\n";
        }

        outFilePCenter << strColumnNames << strValuesParticles.str();

        if(IBM_PARTICLES_NODES_SAVE){
            if((*particles->getPMethod() == IBM)){
                strColumnNames = "particle_index" + sep + "pos_x" + sep + "pos_y" + sep + "pos_z" + sep + "S\n";

                std::ostringstream strValuesMesh("");
                strValuesMesh << std::scientific;

                for(int n_gpu = 0; n_gpu < N_GPUS; n_gpu++){
                    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[n_gpu]));
                    IbmNodesSoA pnSoA = particles->getNodesSoA()[n_gpu];

                    for(int i = 0; i < pnSoA.getNumNodes(); i++){
                        dfloat3 pos = pnSoA.getPos().getValuesFromIdx(i);
                        strValuesMesh << pnSoA.getParticleCenterIdx()[i] << sep;
                        strValuesMesh << getStrDfloat3(pos, sep) << sep;
                        strValuesMesh << pnSoA.getS()[i] << "\n";
                    }
                }

                // Names of file to save particle info
                std::string strFilePNodes = getVarFilename("pNodes", step, ".csv");

                // File to save particle info
                std::ofstream outFilePNodes(strFilePNodes);

                outFilePNodes << strColumnNames << strValuesMesh.str();
            } 
        }
        savingMacrParticle = false;
    });
    saveThread.join();
}


void collectAndExportWallForces(
    ParticleWallForces* d_pwForces,
    unsigned int step
){
    ParticleWallForces h_pwForces;

    cudaMemcpy(&h_pwForces, d_pwForces,sizeof(ParticleWallForces), cudaMemcpyDeviceToHost);

    {
        std::ostringstream s;
        s << "step," << step
          << "," << h_pwForces.bottom.Fx
          << "," << h_pwForces.bottom.Fy
          << "," << h_pwForces.bottom.Fz
          << "," << h_pwForces.bottom.Fn
          << "," << h_pwForces.bottom.Ft
          << "," << h_pwForces.bottom.nContacts;

        saveTreatData("_wall_forces_bottom", s.str(), step);
    }


    {
        std::ostringstream s;
        s << "step," << step
          << "," << h_pwForces.top.Fx
          << "," << h_pwForces.top.Fy
          << "," << h_pwForces.top.Fz
          << "," << h_pwForces.top.Fn
          << "," << h_pwForces.top.Ft
          << "," << h_pwForces.top.nContacts;

        saveTreatData("_wall_forces_top", s.str(), step);
    }
}


#endif //PARTICLE_MODEL