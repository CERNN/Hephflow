int x0 = NX/2;
int z0 = NZ_TOTAL/2;

velocityProfile(h_fMom, 1, x0, 0, z0, step);
phiProfile(h_fMom, 2, x0, 0, z0, step);
conformationProfile(h_fMom, 1, x0, 0, z0, step);

if (step == REPORT_SAVE) {
	std::ostringstream meta;
	meta << std::scientific << std::setprecision(8);
	meta << "step=" << step
		 << "\tmu1=" << MU_1
		 << "\tmu2=" << MU_2
		 << "\teta_s_V=" << ETA_S_V
		 << "\teta_p_V=" << ETA_P_V
		 << "\tlambda_V=" << LAMBDA
		 << "\tbeta=" << BETA
		 << "\ttau1=" << TAU
		 << "\ttau2_total=" << TAU_PHASE2
		 << "\ttau2_solvent=" << TAU_SOLVENT_PHASE2
		 << "\tfx=" << FX
		 << "\tphi_del=" << phi_del;
	saveTreatData("_vv_case007_meta", meta.str(), step, true);
}








