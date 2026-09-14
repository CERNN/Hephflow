AyzVar = AyzVar;

Ayz_qx_t30 = Ayz_qx_t30 + CONF_OMEGA * (AyzVar * ux_t30 - Ayz_qx_t30);
Ayz_qy_t30 = Ayz_qy_t30 + CONF_OMEGA * (AyzVar * uy_t30 - Ayz_qy_t30);
Ayz_qz_t30 = Ayz_qz_t30 + CONF_OMEGA * (AyzVar * uz_t30 - Ayz_qz_t30);
