// ─── Obtain Āxy from the first interior node (pre-streaming populations sum) ───
// You need this from the adjacent interior node, NOT the current boundary node.
// How you load it depends on your data layout. One common approach:
// load the gNode of the neighbour and sum it, then call it AbarInt.
// If you already have access to fMom-style moments of the interior node,
// AbarInt = fMom[interior, A_ZZ_C_INDEX] - Gxy/2  (since A = Ābar + G/2)

// Alternatively, estimate Āxy directly from the known (already-streamed)
// populations at the boundary node before reconstruction:
//   AbarInt = gNode[0]+gNode[1]+...+gNode[18]  (the known ones; missing ones
//             haven't arrived so they still hold the pre-stream value)
// The safest and most common approach is to use the interior neighbour's Ābar.

// Weight constants for clarity
constexpr double W_FACE = 1.0/18.0;   // |ξ|₁ = 1
constexpr double W_EDGE = 1.0/36.0;   // |ξ|₁ = 2

switch (nodeType) {
    case BULK:
        break;

    case NORTH:
    {
        // Interior neighbour is at (y-1): load its Ābar
        // (sum of its gNode populations, not the physical A)
        const double AbarInt1 = fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        const double AbarInt2 = fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        const double AbarInt3 = fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        const double AbarInt = 3.0 * AbarInt1 - 3.0 * AbarInt2 + AbarInt3;

        // Missing populations: ξ_y > 0  →  indices 3,7,11,14,17
        // Known  opposites   : ξ_y < 0  →  indices 4,8,12,13,18
        // Reconstruction: gNode[miss] = -gNode[opp] + 2*w*AbarInt
        gNode[ 4] = -gNode[ 3] + 2.0 * W_FACE * AbarInt;   // ξ=(0,+1, 0)
        gNode[ 8] = -gNode[ 7] + 2.0 * W_EDGE * AbarInt;   // ξ=(+1,+1, 0)
        gNode[12] = -gNode[11] + 2.0 * W_EDGE * AbarInt;   // ξ=(0,+1,+1)
        gNode[13] = -gNode[14] + 2.0 * W_EDGE * AbarInt;   // ξ=(-1,+1, 0)
        gNode[18] = -gNode[17] + 2.0 * W_EDGE * AbarInt;   // ξ=(0,+1,-1)

        // Now recompute macroscopic moments using the STANDARD bulk formulas

                AxyVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] 
                #ifdef D3G19
                + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                #endif 
                #ifdef D3G27
                + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                + gNode[19] + gNode[20] + gNode[21] + gNode[22] + gNode[23] + gNode[24] + gNode[25] + gNode[26]
                #endif
                ;
                AxyVar = AxyVar + Gxy;

                Axy_qx_t30 = (gNode[1] - gNode[2] 
                    #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    #endif
                    #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] + gNode[23] - gNode[24] - gNode[25] + gNode[26]
                    #endif
                );
                Axy_qy_t30 = (gNode[3] - gNode[4]
                    #ifdef D3G19 
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    #endif
                    #ifdef D3G27 
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] - gNode[23] + gNode[24] + gNode[25] - gNode[26]
                    #endif
                );
                Axy_qz_t30 = (gNode[5] - gNode[6]
                    #ifdef D3G19 
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    #endif
                    #ifdef D3G27 
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    + gNode[19] - gNode[20] - gNode[21] + gNode[22] + gNode[23] - gNode[24] + gNode[25] - gNode[26]
                    #endif
                );
        break;
    }

    case SOUTH:
    {
        // Interior neighbour is at (y+1)
        const double AbarInt1 = fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        const double AbarInt2 = fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        const double AbarInt3 = fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        const double AbarInt = 3.0 * AbarInt1 - 3.0 * AbarInt2 + AbarInt3;

        // Missing: ξ_y < 0  →  4,8,12,13,18;  opp: 3,7,11,14,17
        gNode[ 3] = -gNode[ 4] + 2.0 * W_FACE * AbarInt;   // ξ=(0,-1, 0)
        gNode[ 7] = -gNode[ 8] + 2.0 * W_EDGE * AbarInt;   // ξ=(-1,-1, 0)
        gNode[11] = -gNode[12] + 2.0 * W_EDGE * AbarInt;   // ξ=(0,-1,-1)
        gNode[14] = -gNode[13] + 2.0 * W_EDGE * AbarInt;   // ξ=(+1,-1, 0)
        gNode[17] = -gNode[18] + 2.0 * W_EDGE * AbarInt;   // ξ=(0,-1,+1)


        // Recompute moments — identical bulk formulas

                AxyVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] 
                #ifdef D3G19
                + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                #endif 
                #ifdef D3G27
                + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                + gNode[19] + gNode[20] + gNode[21] + gNode[22] + gNode[23] + gNode[24] + gNode[25] + gNode[26]
                #endif
                ;
                AxyVar = AxyVar + Gxy;

                Axy_qx_t30 = (gNode[1] - gNode[2] 
                    #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    #endif
                    #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] + gNode[23] - gNode[24] - gNode[25] + gNode[26]
                    #endif
                );
                Axy_qy_t30 = (gNode[3] - gNode[4]
                    #ifdef D3G19 
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    #endif
                    #ifdef D3G27 
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] - gNode[23] + gNode[24] + gNode[25] - gNode[26]
                    #endif
                );
                Axy_qz_t30 = (gNode[5] - gNode[6]
                    #ifdef D3G19 
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    #endif
                    #ifdef D3G27 
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    + gNode[19] - gNode[20] - gNode[21] + gNode[22] + gNode[23] - gNode[24] + gNode[25] - gNode[26]
                    #endif
                );
        break;
    }

    default:
        break;
}