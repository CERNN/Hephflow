rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

#ifdef PHI_DIST
/*
dfloat R = 30.0; // mean radius
dfloat CHI = 2.0; // interface parameter
dfloat eps_amp = 0.05;   // perturbation amplitude

const dfloat inv_sqrt2_CHI = 1.0 / (sqrt(2.0) * CHI);
const dfloat phi_mid = 0.5 * (PHI_ONE + PHI_TWO);
const dfloat phi_diff = 0.5 * (PHI_TWO - PHI_ONE);

dfloat dx = x - (dfloat)(NX-1)/2;
dfloat dy = y - (dfloat)(NY-1)/2;
dfloat dz = z - (dfloat)(NZ-1)/2;

dfloat r = sqrtf(dx*dx + dy*dy + dz*dz);
dfloat cos_theta = (r < 1e-14) ? 1.0 : dz / r;


dfloat P2 = 0.5 * (3.0 * cos_theta * cos_theta - 1.0); 
dfloat Rp = R * (1.0 + eps_amp * P2); 
dfloat s = r - Rp; 

dfloat phi_raw = phi_mid + phi_diff * tanh(-s * inv_sqrt2_CHI); //Smooth phase field

if (PHI_TWO > PHI_ONE){
    if (phi_raw < PHI_ONE) phi_raw = PHI_ONE;
    if (phi_raw > PHI_TWO) phi_raw = PHI_TWO;
} else {
    if (phi_raw < PHI_TWO) phi_raw = PHI_TWO;
    if (phi_raw > PHI_ONE) phi_raw = PHI_ONE;
}

phiVar = phi_raw;
*/

const dfloat phi_mid  = 0.5 * (PHI_ONE + PHI_TWO);
const dfloat phi_half = 0.5 * (PHI_TWO - PHI_ONE);

const dfloat a = 50.0;
const dfloat b = 15.0;
const dfloat c = 10.0;
const dfloat xi = 1.0;

const dfloat denom = sqrt(2.0) * xi;

dfloat X = (dfloat)x - (NX-1)/2.0;
dfloat Y = (dfloat)y - (NY-1)/2.0;
dfloat Z = (dfloat)z - (NZ-1)/2.0;

dfloat f = X*X/(a*a) + Y*Y/(b*b) + Z*Z/(c*c) - 1.0;

// gradient of f
dfloat gx = 2.0 * X / (a*a);
dfloat gy = 2.0 * Y / (b*b);
dfloat gz = 2.0 * Z / (c*c);

dfloat gnorm2 = gx*gx + gy*gy + gz*gz;
dfloat gnorm = sqrt(gnorm2 + 1e-20);

dfloat delta = f / gnorm;  //signed distance ( >0 outside, <0 inside )

phiVar = phi_mid + phi_half * tanh( -delta / denom );

#endif

