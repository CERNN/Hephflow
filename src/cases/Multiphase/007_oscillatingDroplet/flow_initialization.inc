rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;


#ifdef PHI_DIST

const dfloat phi_mid  = 0.5 * (PHI_ONE + PHI_TWO);
const dfloat phi_half = 0.5 * (PHI_TWO - PHI_ONE);

const dfloat a = 20.0;
const dfloat b = 20.0;
const dfloat c = 30.0;

const dfloat denom = sqrt(2.0) * phi_xi;

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
phiVar += PHI_ZERO;

#endif

/*

#ifdef PHI_DIST

const dfloat phi_mid  = 0.5 * (PHI_ONE + PHI_TWO);
const dfloat phi_half = 0.5 * (PHI_TWO - PHI_ONE);
const dfloat denom    = sqrt(2.0) * phi_xi;

// --- Bubble Parameters ---
const dfloat R = 15.0_df;           // Radius
const dfloat gap = R * 2.2_df;      // Distance between centers (2.2R = 0.2R gap)

// Center of the domain
const dfloat midX = (NX-1)/2.0_df;
const dfloat midY = (NY-1)/2.0_df;
const dfloat midZ = (NZ-1)/2.0_df;

// --- Center Positions (forming an L) ---
// Bubble 1: The Corner (shifted so the L is centered)
dfloat cx1 = midX - gap/3.0_df; 
dfloat cy1 = midY - gap/3.0_df; 
dfloat cz1 = midZ;

// Bubble 2: Offset in X
dfloat cx2 = cx1 + gap;
dfloat cy2 = cy1;
dfloat cz2 = cz1;

// Bubble 3: Offset in Y
dfloat cx3 = cx1;
dfloat cy3 = cy1 + gap;
dfloat cz3 = cz1;

// --- Distance Calculations ---
dfloat dx1 = (dfloat)x - cx1;
dfloat dy1 = (dfloat)y - cy1;
dfloat dz1 = (dfloat)z - cz1;
dfloat dist1 = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1) - R;

dfloat dx2 = (dfloat)x - cx2;
dfloat dy2 = (dfloat)y - cy2;
dfloat dz2 = (dfloat)z - cz2;
dfloat dist2 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2) - R;

dfloat dx3 = (dfloat)x - cx3;
dfloat dy3 = (dfloat)y - cy3;
dfloat dz3 = (dfloat)z - cz3;
dfloat dist3 = sqrt(dx3*dx3 + dy3*dy3 + dz3*dz3) - R;

// Union of all three spheres
dfloat delta = fminf(dist1, fminf(dist2, dist3));

// Apply the tanh profile
phiVar = phi_mid + phi_half * tanh( -delta / denom );
phiVar += PHI_ZERO;

#endif
*/