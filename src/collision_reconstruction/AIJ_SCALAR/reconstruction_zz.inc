gNode[ 0] = AzzVar * gW0;
pics2 = 1.0_df;

multiplyTerm = AzzVar * gW1;
gNode[ 1] = gW1 * (AzzVar + Azz_qx_t30);
gNode[ 2] = gW1 * (AzzVar - Azz_qx_t30);
gNode[ 3] = gW1 * (AzzVar + Azz_qy_t30);
gNode[ 4] = gW1 * (AzzVar - Azz_qy_t30);
gNode[ 5] = gW1 * (AzzVar + Azz_qz_t30);
gNode[ 6] = gW1 * (AzzVar - Azz_qz_t30);
#ifdef D3G19
multiplyTerm = AzzVar * gW2;
gNode[ 7] = gW2 * (AzzVar + Azz_qx_t30 + Azz_qy_t30);
gNode[ 8] = gW2 * (AzzVar - Azz_qx_t30 - Azz_qy_t30);
gNode[ 9] = gW2 * (AzzVar + Azz_qx_t30 + Azz_qz_t30);
gNode[10] = gW2 * (AzzVar - Azz_qx_t30 - Azz_qz_t30);
gNode[11] = gW2 * (AzzVar + Azz_qy_t30 + Azz_qz_t30);
gNode[12] = gW2 * (AzzVar - Azz_qy_t30 - Azz_qz_t30);
gNode[13] = gW2 * (AzzVar + Azz_qx_t30 - Azz_qy_t30);
gNode[14] = gW2 * (AzzVar - Azz_qx_t30 + Azz_qy_t30);
gNode[15] = gW2 * (AzzVar + Azz_qx_t30 - Azz_qz_t30);
gNode[16] = gW2 * (AzzVar - Azz_qx_t30 + Azz_qz_t30);
gNode[17] = gW2 * (AzzVar + Azz_qy_t30 - Azz_qz_t30);
gNode[18] = gW2 * (AzzVar - Azz_qy_t30 + Azz_qz_t30);
#endif
#ifdef D3G27
multiplyTerm = AzzVar * gW3;
gNode[ 7] = gW2 * (AzzVar + Azz_qx_t30 + Azz_qy_t30);
gNode[ 8] = gW2 * (AzzVar - Azz_qx_t30 - Azz_qy_t30);
gNode[ 9] = gW2 * (AzzVar + Azz_qx_t30 + Azz_qz_t30);
gNode[10] = gW2 * (AzzVar - Azz_qx_t30 - Azz_qz_t30);
gNode[11] = gW2 * (AzzVar + Azz_qy_t30 + Azz_qz_t30);
gNode[12] = gW2 * (AzzVar - Azz_qy_t30 - Azz_qz_t30);
gNode[13] = gW2 * (AzzVar + Azz_qx_t30 - Azz_qy_t30);
gNode[14] = gW2 * (AzzVar - Azz_qx_t30 + Azz_qy_t30);
gNode[15] = gW2 * (AzzVar + Azz_qx_t30 - Azz_qz_t30);
gNode[16] = gW2 * (AzzVar - Azz_qx_t30 + Azz_qz_t30);
gNode[17] = gW2 * (AzzVar + Azz_qy_t30 - Azz_qz_t30);
gNode[18] = gW2 * (AzzVar - Azz_qy_t30 + Azz_qz_t30);
multiplyTerm = AzzVar * gW3;
gNode[19] = gW3 * (AzzVar + Azz_qx_t30 + Azz_qy_t30 + Azz_qz_t30);
gNode[20] = gW3 * (AzzVar - Azz_qx_t30 - Azz_qy_t30 - Azz_qz_t30);
gNode[21] = gW3 * (AzzVar + Azz_qx_t30 + Azz_qy_t30 - Azz_qz_t30);
gNode[22] = gW3 * (AzzVar - Azz_qx_t30 - Azz_qy_t30 + Azz_qz_t30);
gNode[23] = gW3 * (AzzVar + Azz_qx_t30 - Azz_qy_t30 + Azz_qz_t30);
gNode[24] = gW3 * (AzzVar - Azz_qx_t30 + Azz_qy_t30 - Azz_qz_t30);
gNode[25] = gW3 * (AzzVar - Azz_qx_t30 + Azz_qy_t30 + Azz_qz_t30);
gNode[26] = gW3 * (AzzVar + Azz_qx_t30 - Azz_qy_t30 - Azz_qz_t30);
#endif