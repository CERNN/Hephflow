phiVar = phiVar;

phi_qx_t30 = phi_qx_t30 + PHI_OMEGA * (phiVar * ux_t30 - phi_qx_t30);
phi_qy_t30 = phi_qy_t30 + PHI_OMEGA * (phiVar * uy_t30 - phi_qy_t30);
phi_qz_t30 = phi_qz_t30 + PHI_OMEGA * (phiVar * uz_t30 - phi_qz_t30);
