// Track RTI front coordinates at the three canonical points of the single-mode perturbation.
// Bubble: (x=L/2,z=L/2), Spike: (x=0,z=0), Saddle: (x=0,z=L/2).
{
	auto samplePhiAt = [&](int x, int y, int z) -> dfloat {
		dfloat phi = 0.0_df;
		const size_t idx = idxMom(
			x % BLOCK_NX,
			y % BLOCK_NY,
			z % BLOCK_NZ,
			M3_PHI_INDEX,
			x / BLOCK_NX,
			y / BLOCK_NY,
			z / BLOCK_NZ
		);
		checkCudaErrors(cudaMemcpy(&phi, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
		return phi;
	};

	auto findInterfaceYAtXZ = [&](int xProbe, int zProbe, dfloat phiIso) -> dfloat {
		const dfloat yMid = 0.5_df * (dfloat)(NY - 1);
		bool found = false;
		dfloat bestY = yMid;
		dfloat bestDist = 1.0e30_df;

		dfloat prev = samplePhiAt(xProbe, 0, zProbe);
		for (int y = 1; y < NY; ++y) {
			const dfloat curr = samplePhiAt(xProbe, y, zProbe);
			const dfloat s0 = prev - phiIso;
			const dfloat s1 = curr - phiIso;

			if ((s0 == 0.0_df) || (s0 * s1 <= 0.0_df)) {
				dfloat yCross = (dfloat)(y - 1);
				const dfloat denom = curr - prev;
				if (fabs(denom) > 1.0e-12_df) {
					yCross += (phiIso - prev) / denom;
				}

				const dfloat distMid = fabs(yCross - yMid);
				if (!found || distMid < bestDist) {
					found = true;
					bestDist = distMid;
					bestY = yCross;
				}
			}

			prev = curr;
		}

		return bestY;
	};

	const int bubbleX = NX / 2;
	const int bubbleZ = NZ / 2;
	const int spikeX = 0;
	const int spikeZ = 0;
	const int saddleX = 0;
	const int saddleZ = NZ / 2;

	const dfloat bubbleY_idx = findInterfaceYAtXZ(bubbleX, bubbleZ, PHI_REFERENCE);
	const dfloat spikeY_idx = findInterfaceYAtXZ(spikeX, spikeZ, PHI_REFERENCE);
	const dfloat saddleY_idx = findInterfaceYAtXZ(saddleX, saddleZ, PHI_REFERENCE);

	const dfloat yCenter = 0.5_df * (dfloat)NY;
	const dfloat bubbleY = (bubbleY_idx - yCenter) / (dfloat)L;
	const dfloat spikeY  = (spikeY_idx - yCenter) / (dfloat)L;
	const dfloat saddleY = (saddleY_idx - yCenter) / (dfloat)L;

	const dfloat tScale = sqrt((dfloat)L / fabs(g_value));
	const dfloat tStar = (tScale > 0.0_df) ? ((dfloat)step / tScale) : 0.0_df;

	const std::filesystem::path baseDir = folderSetup();
	const std::filesystem::path csvPath = baseDir / "rti_front_tracking.csv";

	if (step == REPORT_SAVE) {
		std::ofstream header(csvPath, std::ios::trunc);
		if (header.is_open()) {
			header << "t_star,bubble_y,spike_y,saddle_y" << std::endl;
		}
	}

	std::ofstream out(csvPath, std::ios::app);
	if (out.is_open()) {
		out << std::scientific << std::setprecision(8)
			<< tStar << ","
			<< bubbleY << ","
			<< spikeY << ","
			<< saddleY << std::endl;
	} else {
		std::cerr << "[RTI treat_data] ERROR: failed to open output file: " << csvPath << std::endl;
	}
}
