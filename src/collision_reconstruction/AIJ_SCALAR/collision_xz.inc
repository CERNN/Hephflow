AxzVar = AxzVar;

Axz_qx_t30 = Axz_qx_t30 + CONF_OMEGA * (AxzVar * ux_t30 - Axz_qx_t30);
Axz_qy_t30 = Axz_qy_t30 + CONF_OMEGA * (AxzVar * uy_t30 - Axz_qy_t30);
Axz_qz_t30 = Axz_qz_t30 + CONF_OMEGA * (AxzVar * uz_t30 - Axz_qz_t30);
