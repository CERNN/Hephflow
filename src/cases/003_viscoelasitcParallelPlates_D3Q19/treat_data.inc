const int x0 = NX / 2;
const int z0 = (NZ/N_GPUS) / 2;
velocityProfile(h_fMom, 1, x0, 0, z0, step);
conformationProfile(h_fMom, 1, x0, 0, z0, step);
