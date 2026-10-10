/* Trajectory files are emitted by PARTICLES_SAVE every 100 steps. */
#ifdef TESTS
if (step == N_STEPS - 1) {
    checkCudaErrors(cudaStreamSynchronize(particleField.getStream()[0]));
    const ParticleCenter particle =
        particleField.getParticlesSoA()->getPCenterArray()[0];
    const dfloat3 pos = particle.getPos();
    const dfloat3 vel = particle.getVel();
    FILE* test_file = fopen("test_results.txt", "w");
    if (test_file != NULL) {
        ParticleCenter* centers = particleField.getParticlesSoA()->getPCenterArray();
        const dfloat3 pos = centers[0].getPos();
        const dfloat3 vel = centers[0].getVel();
        fprintf(test_file, "STEP,%d\n", step);
        fprintf(test_file, "FINAL_POS_Y,%.15e\n", pos.y);
        fprintf(test_file, "FINAL_VEL_Y,%.15e\n", vel.y);
        fprintf(test_file, "FINITE_STATE,%d\n",
            isfinite(pos.y) && isfinite(vel.y) ? 1 : 0);
        fclose(test_file);
        printf("[TEST] Settling trajectory ready for validation\n");
    }
}
#endif
