// Race-free physical-A and stored-representation neighbors. The auxiliary
// scalar LBM diffuses the stored tensor B. Transformed formulations use the
// two Laplacians below to replace D laplacian(B) by the inverse differential
// image of D laplacian(A), so all representations solve the same regularized
// physical conformation equation.

const Sym3 veDirectStored(
    fMom[idxMom(tx,ty,tz,A_XX_C_INDEX,bx,by,bz)],
    fMom[idxMom(tx,ty,tz,A_XY_C_INDEX,bx,by,bz)],
    fMom[idxMom(tx,ty,tz,A_XZ_C_INDEX,bx,by,bz)],
    fMom[idxMom(tx,ty,tz,A_YY_C_INDEX,bx,by,bz)],
    fMom[idxMom(tx,ty,tz,A_YZ_C_INDEX,bx,by,bz)],
    fMom[idxMom(tx,ty,tz,A_ZZ_C_INDEX,bx,by,bz)]);

#if CONF_REPRESENTATION == CONF_REP_LOG
VeEigenSym3 veDirectPsiEigen;
const Sym3 veDirectPhysicalA=veExpSymmetric(veDirectStored,&veDirectPsiEigen);
#else
const Sym3 veDirectPhysicalA=vePhysicalAFromStored(veDirectStored);
#endif // CONF_REPRESENTATION == CONF_REP_LOG

const int veDirectLocalIndex=tx+BLOCK_NX*(ty+BLOCK_NY*tz);

#define STAGE_SYM3(offset,value) \
    do { \
        s_pop[veDirectLocalIndex+((offset)+0)*BLOCK_LBM_SIZE]=(value).xx; \
        s_pop[veDirectLocalIndex+((offset)+1)*BLOCK_LBM_SIZE]=(value).xy; \
        s_pop[veDirectLocalIndex+((offset)+2)*BLOCK_LBM_SIZE]=(value).xz; \
        s_pop[veDirectLocalIndex+((offset)+3)*BLOCK_LBM_SIZE]=(value).yy; \
        s_pop[veDirectLocalIndex+((offset)+4)*BLOCK_LBM_SIZE]=(value).yz; \
        s_pop[veDirectLocalIndex+((offset)+5)*BLOCK_LBM_SIZE]=(value).zz; \
    } while(false)

STAGE_SYM3(0,veDirectPhysicalA);
STAGE_SYM3(6,veDirectStored);
activeBarrier.arrive_and_wait();

const int veXm=veDirectLocalIndex-1, veXp=veDirectLocalIndex+1;
const int veYm=veDirectLocalIndex-BLOCK_NX, veYp=veDirectLocalIndex+BLOCK_NX;
const int veZm=veDirectLocalIndex-BLOCK_NX*BLOCK_NY, veZp=veDirectLocalIndex+BLOCK_NX*BLOCK_NY;

#define LOAD_DIRECTION_NEIGHBOR(direction,side,sharedCondition,sharedIndex,haloFace,haloIndex) \
    Sym3 veA##direction##side; \
    Sym3 veB##direction##side; \
    if(sharedCondition) { \
        veA##direction##side=Sym3( \
            s_pop[(sharedIndex)+0*BLOCK_LBM_SIZE],s_pop[(sharedIndex)+1*BLOCK_LBM_SIZE], \
            s_pop[(sharedIndex)+2*BLOCK_LBM_SIZE],s_pop[(sharedIndex)+3*BLOCK_LBM_SIZE], \
            s_pop[(sharedIndex)+4*BLOCK_LBM_SIZE],s_pop[(sharedIndex)+5*BLOCK_LBM_SIZE]); \
        veB##direction##side=Sym3( \
            s_pop[(sharedIndex)+6*BLOCK_LBM_SIZE],s_pop[(sharedIndex)+7*BLOCK_LBM_SIZE], \
            s_pop[(sharedIndex)+8*BLOCK_LBM_SIZE],s_pop[(sharedIndex)+9*BLOCK_LBM_SIZE], \
            s_pop[(sharedIndex)+10*BLOCK_LBM_SIZE],s_pop[(sharedIndex)+11*BLOCK_LBM_SIZE]); \
    } else { \
        const size_t veHaloBase=(haloIndex); \
        veA##direction##side=Sym3( \
            (haloFace)[veHaloBase+0],(haloFace)[veHaloBase+1],(haloFace)[veHaloBase+2], \
            (haloFace)[veHaloBase+3],(haloFace)[veHaloBase+4],(haloFace)[veHaloBase+5]); \
        veB##direction##side=Sym3( \
            (haloFace)[veHaloBase+6],(haloFace)[veHaloBase+7],(haloFace)[veHaloBase+8], \
            (haloFace)[veHaloBase+9],(haloFace)[veHaloBase+10],(haloFace)[veHaloBase+11]); \
    }

LOAD_DIRECTION_NEIGHBOR(X,m,x>0 && tx>0,veXm,polymerStressHalo.X_1,
    g_idxPolymerStressHaloX(ty,tz,0,bxm1,by,bz));
LOAD_DIRECTION_NEIGHBOR(X,p,x+1<NX && tx<BLOCK_NX-1,veXp,polymerStressHalo.X_0,
    g_idxPolymerStressHaloX(ty,tz,0,bxp1,by,bz));
LOAD_DIRECTION_NEIGHBOR(Y,m,y>0 && ty>0,veYm,polymerStressHalo.Y_1,
    g_idxPolymerStressHaloY(tx,tz,0,bx,bym1,bz));
LOAD_DIRECTION_NEIGHBOR(Y,p,y+1<NY && ty<BLOCK_NY-1,veYp,polymerStressHalo.Y_0,
    g_idxPolymerStressHaloY(tx,tz,0,bx,byp1,bz));
LOAD_DIRECTION_NEIGHBOR(Z,m,z_local>0 && tz>0,veZm,
    z_local==0 ? polymerStressHalo.auxZ_1 : polymerStressHalo.Z_1,
    z_local==0 ? g_idxPolymerStressHaloZ(tx,ty,0,bx,by,0)
               : g_idxPolymerStressHaloZ(tx,ty,0,bx,by,bzm1));
LOAD_DIRECTION_NEIGHBOR(Z,p,z_local+1<localNZ && tz<BLOCK_NZ-1,veZp,
    z_local+1>=localNZ ? polymerStressHalo.auxZ_0 : polymerStressHalo.Z_0,
    z_local+1>=localNZ ? g_idxPolymerStressHaloZ(tx,ty,0,bx,by,0)
                       : g_idxPolymerStressHaloZ(tx,ty,0,bx,by,bzp1));

#undef LOAD_DIRECTION_NEIGHBOR
#undef STAGE_SYM3

Sym3 veLaplacianPhysicalA=veAXm+veAXp+veAYm+veAYp+veAZm+veAZp-veDirectPhysicalA*6.0_df;
Sym3 veLaplacianStored=veBXm+veBXp+veBYm+veBYp+veBZm+veBZp-veDirectStored*6.0_df;

#define MIRROR_LAPLACIAN(mask,exteriorA,interiorA,exteriorB,interiorB) \
    if((nodeType&(mask))==(mask)) { \
        veLaplacianPhysicalA=veLaplacianPhysicalA+(interiorA)-(exteriorA); \
        veLaplacianStored=veLaplacianStored+(interiorB)-(exteriorB); \
    }

MIRROR_LAPLACIAN(0b01010101,veAXm,veAXp,veBXm,veBXp);
MIRROR_LAPLACIAN(0b10101010,veAXp,veAXm,veBXp,veBXm);
MIRROR_LAPLACIAN(0b00110011,veAYm,veAYp,veBYm,veBYp);
MIRROR_LAPLACIAN(0b11001100,veAYp,veAYm,veBYp,veBYm);
MIRROR_LAPLACIAN(0b00001111,veAZm,veAZp,veBZm,veBZp);
MIRROR_LAPLACIAN(0b11110000,veAZp,veAZm,veBZp,veBZm);
#undef MIRROR_LAPLACIAN

#ifdef CONFORMATION_DIRECT_STRESS_HALO
const Sym3 veS0=phasePropsA.ve.type==VE_OLDROYD_B ? veDirectPhysicalA
    : calcVeStressReduced(phasePropsA.ve,veDirectPhysicalA);
const Sym3 veSXm=phasePropsA.ve.type==VE_OLDROYD_B ? veAXm : calcVeStressReduced(phasePropsA.ve,veAXm);
const Sym3 veSXp=phasePropsA.ve.type==VE_OLDROYD_B ? veAXp : calcVeStressReduced(phasePropsA.ve,veAXp);
const Sym3 veSYm=phasePropsA.ve.type==VE_OLDROYD_B ? veAYm : calcVeStressReduced(phasePropsA.ve,veAYm);
const Sym3 veSYp=phasePropsA.ve.type==VE_OLDROYD_B ? veAYp : calcVeStressReduced(phasePropsA.ve,veAYp);
const Sym3 veSZm=phasePropsA.ve.type==VE_OLDROYD_B ? veAZm : calcVeStressReduced(phasePropsA.ve,veAZm);
const Sym3 veSZp=phasePropsA.ve.type==VE_OLDROYD_B ? veAZp : calcVeStressReduced(phasePropsA.ve,veAZp);

dfloat direct_dSxxdx=0.5_df*(veSXp.xx-veSXm.xx), direct_dSxydx=0.5_df*(veSXp.xy-veSXm.xy);
dfloat direct_dSxzdx=0.5_df*(veSXp.xz-veSXm.xz), direct_dSxydy=0.5_df*(veSYp.xy-veSYm.xy);
dfloat direct_dSyydy=0.5_df*(veSYp.yy-veSYm.yy), direct_dSyzdy=0.5_df*(veSYp.yz-veSYm.yz);
dfloat direct_dSxzdz=0.5_df*(veSZp.xz-veSZm.xz), direct_dSyzdz=0.5_df*(veSZp.yz-veSZm.yz);
dfloat direct_dSzzdz=0.5_df*(veSZp.zz-veSZm.zz);

if((nodeType&0b01010101)==0b01010101) {
    direct_dSxxdx=veSXp.xx-veS0.xx; direct_dSxydx=veSXp.xy-veS0.xy; direct_dSxzdx=veSXp.xz-veS0.xz;
}
if((nodeType&0b10101010)==0b10101010) {
    direct_dSxxdx=veS0.xx-veSXm.xx; direct_dSxydx=veS0.xy-veSXm.xy; direct_dSxzdx=veS0.xz-veSXm.xz;
}
if((nodeType&0b00110011)==0b00110011) {
    direct_dSxydy=veSYp.xy-veS0.xy; direct_dSyydy=veSYp.yy-veS0.yy; direct_dSyzdy=veSYp.yz-veS0.yz;
}
if((nodeType&0b11001100)==0b11001100) {
    direct_dSxydy=veS0.xy-veSYm.xy; direct_dSyydy=veS0.yy-veSYm.yy; direct_dSyzdy=veS0.yz-veSYm.yz;
}
if((nodeType&0b00001111)==0b00001111) {
    direct_dSxzdz=veSZp.xz-veS0.xz; direct_dSyzdz=veSZp.yz-veS0.yz; direct_dSzzdz=veSZp.zz-veS0.zz;
}
if((nodeType&0b11110000)==0b11110000) {
    direct_dSxzdz=veS0.xz-veSZm.xz; direct_dSyzdz=veS0.yz-veSZm.yz; direct_dSzzdz=veS0.zz-veSZm.zz;
}

const dfloat directDivStressX=direct_dSxxdx+direct_dSxydy+direct_dSxzdz;
const dfloat directDivStressY=direct_dSxydx+direct_dSyydy+direct_dSyzdz;
const dfloat directDivStressZ=direct_dSxzdx+direct_dSyzdy+direct_dSzzdz;
#endif // CONFORMATION_DIRECT_STRESS_HALO

activeBarrier.arrive_and_wait();
