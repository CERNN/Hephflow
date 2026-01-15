// s = beta nabla .(phi(1-phi)n)
// s = beta [nabla (phi(1-phi)).n + phi(1-phi)nabla .n]
// s = beta [(1-2phi)|nabla phi| + phi(1-phi)kappa]

dfloat phiSharpSource = 0.0_df;

dfloat phi_factor = phi_for_sharp * (1.0_df - phi_for_sharp);  // phi(1-phi)
dfloat sign_term = 1.0_df - 2.0_df * phi_for_sharp ;  // (1-2phi)

// (1-2phi)|nabla phi|
dfloat sharp_antidiff = sign_term * mod_grad;

// phi(1-phi)kappa
dfloat sharp_curv = phi_factor * curvature;

// s = beta [(1-2phi)|nabla phi| + phi(1-phi)kappa]
phiSharpSource = - phi_sharp_c * (sharp_antidiff - sharp_curv);
