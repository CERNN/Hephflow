gNode[ 0] = AyyVar * gW0;
pics2 = 1.0_df;

multiplyTerm = AyyVar * gW1;
gNode[ 1] = gW1 * (AyyVar + Ayy_qx_t30);
gNode[ 2] = gW1 * (AyyVar - Ayy_qx_t30);
gNode[ 3] = gW1 * (AyyVar + Ayy_qy_t30);
gNode[ 4] = gW1 * (AyyVar - Ayy_qy_t30);
gNode[ 5] = gW1 * (AyyVar + Ayy_qz_t30);
gNode[ 6] = gW1 * (AyyVar - Ayy_qz_t30);
#ifdef D3G19
multiplyTerm = AyyVar * gW2;
gNode[ 7] = gW2 * (AyyVar + Ayy_qx_t30 + Ayy_qy_t30);
gNode[ 8] = gW2 * (AyyVar - Ayy_qx_t30 - Ayy_qy_t30);
gNode[ 9] = gW2 * (AyyVar + Ayy_qx_t30 + Ayy_qz_t30);
gNode[10] = gW2 * (AyyVar - Ayy_qx_t30 - Ayy_qz_t30);
gNode[11] = gW2 * (AyyVar + Ayy_qy_t30 + Ayy_qz_t30);
gNode[12] = gW2 * (AyyVar - Ayy_qy_t30 - Ayy_qz_t30);
gNode[13] = gW2 * (AyyVar + Ayy_qx_t30 - Ayy_qy_t30);
gNode[14] = gW2 * (AyyVar - Ayy_qx_t30 + Ayy_qy_t30);
gNode[15] = gW2 * (AyyVar + Ayy_qx_t30 - Ayy_qz_t30);
gNode[16] = gW2 * (AyyVar - Ayy_qx_t30 + Ayy_qz_t30);
gNode[17] = gW2 * (AyyVar + Ayy_qy_t30 - Ayy_qz_t30);
gNode[18] = gW2 * (AyyVar - Ayy_qy_t30 + Ayy_qz_t30);
#endif
#ifdef D3G27
multiplyTerm = AyyVar * gW3;
gNode[ 7] = gW2 * (AyyVar + Ayy_qx_t30 + Ayy_qy_t30);
gNode[ 8] = gW2 * (AyyVar - Ayy_qx_t30 - Ayy_qy_t30);
gNode[ 9] = gW2 * (AyyVar + Ayy_qx_t30 + Ayy_qz_t30);
gNode[10] = gW2 * (AyyVar - Ayy_qx_t30 - Ayy_qz_t30);
gNode[11] = gW2 * (AyyVar + Ayy_qy_t30 + Ayy_qz_t30);
gNode[12] = gW2 * (AyyVar - Ayy_qy_t30 - Ayy_qz_t30);
gNode[13] = gW2 * (AyyVar + Ayy_qx_t30 - Ayy_qy_t30);
gNode[14] = gW2 * (AyyVar - Ayy_qx_t30 + Ayy_qy_t30);
gNode[15] = gW2 * (AyyVar + Ayy_qx_t30 - Ayy_qz_t30);
gNode[16] = gW2 * (AyyVar - Ayy_qx_t30 + Ayy_qz_t30);
gNode[17] = gW2 * (AyyVar + Ayy_qy_t30 - Ayy_qz_t30);
gNode[18] = gW2 * (AyyVar - Ayy_qy_t30 + Ayy_qz_t30);
multiplyTerm = AyyVar * gW3;
gNode[19] = gW3 * (AyyVar + Ayy_qx_t30 + Ayy_qy_t30 + Ayy_qz_t30);
gNode[20] = gW3 * (AyyVar - Ayy_qx_t30 - Ayy_qy_t30 - Ayy_qz_t30);
gNode[21] = gW3 * (AyyVar + Ayy_qx_t30 + Ayy_qy_t30 - Ayy_qz_t30);
gNode[22] = gW3 * (AyyVar - Ayy_qx_t30 - Ayy_qy_t30 + Ayy_qz_t30);
gNode[23] = gW3 * (AyyVar + Ayy_qx_t30 - Ayy_qy_t30 + Ayy_qz_t30);
gNode[24] = gW3 * (AyyVar - Ayy_qx_t30 + Ayy_qy_t30 - Ayy_qz_t30);
gNode[25] = gW3 * (AyyVar - Ayy_qx_t30 + Ayy_qy_t30 + Ayy_qz_t30);
gNode[26] = gW3 * (AyyVar + Ayy_qx_t30 - Ayy_qy_t30 - Ayy_qz_t30);
#endif