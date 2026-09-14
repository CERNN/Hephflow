AxyVar = AxyVar;

Axy_qx_t30 = Axy_qx_t30 + CONF_OMEGA * (AxyVar * ux_t30 - Axy_qx_t30);
Axy_qy_t30 = Axy_qy_t30 + CONF_OMEGA * (AxyVar * uy_t30 - Axy_qy_t30);
Axy_qz_t30 = Axy_qz_t30 + CONF_OMEGA * (AxyVar * uz_t30 - Axy_qz_t30);
