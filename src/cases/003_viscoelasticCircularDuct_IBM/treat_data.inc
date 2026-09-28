// Fully developed circular Oldroyd-B/IBM validation over all physical nodes.
static_assert(DUCT_RADIUS_A == DUCT_RADIUS_B,
    "The analytical validation below is for a circular duct.");

auto ductMoment = [&](int px, int py, int pz, int moment) -> double {
    return static_cast<double>(h_fMom[idxMom(
        px % BLOCK_NX, py % BLOCK_NY, pz % BLOCK_NZ, moment,
        px / BLOCK_NX, py / BLOCK_NY, pz / BLOCK_NZ)]);
};
auto ductUz = [&](int px, int py, int pz) -> double {
    return ductMoment(px, py, pz, M_UZ_INDEX) /
        static_cast<double>(F_M_I_SCALE);
};
auto ductA = [&](int px, int py, int pz, int moment) -> double {
    return ductMoment(px, py, pz, moment) - static_cast<double>(CONF_ZERO);
};

const int aMoments[6] = {
    A_XX_C_INDEX, A_XY_C_INDEX, A_XZ_C_INDEX,
    A_YY_C_INDEX, A_YZ_C_INDEX, A_ZZ_C_INDEX
};
const char* aNames[6] = {"Axx", "Axy", "Axz", "Ayy", "Ayz", "Azz"};
constexpr int ductBins = static_cast<int>(DUCT_RADIUS) + 1;
double radiusSum[ductBins] = {};
double uzSum[ductBins] = {};
double uzExactSum[ductBins] = {};
double urSum[ductBins] = {};
double arzSum[ductBins] = {};
double arzExactSum[ductBins] = {};
double azzSum[ductBins] = {};
double azzExactSum[ductBins] = {};
double tauPrzSum[ductBins] = {};
double n1Sum[ductBins] = {};
unsigned long long binCount[ductBins] = {};

// Region order: complete physical pipe, bulk, first two near-wall layers.
double velocityErrorSq[3] = {}, velocityExactSq[3] = {}, velocityLinf[3] = {};
double aErrorSq[6][3] = {}, aExactSq[6][3] = {}, aLinf[6][3] = {};
unsigned long long regionCount[3] = {};
double flowRateSum = 0.0;
double maxRadialVelocity = 0.0;
double maxNearWallShearError = 0.0;
double minimumPrincipalMinor = 1.0e300;
double polymerForceSum = 0.0;
double polymerForceErrorSq = 0.0;
unsigned long long polymerForceCount = 0;

const double radius = static_cast<double>(DUCT_RADIUS);
const double radiusSquared = radius * radius;
const double eta0 = static_cast<double>(RHO_0 * SUM_VISC);
const double etaP = static_cast<double>(RHO_0 * nu_p);
const double lambda = static_cast<double>(LAMBDA);
const double driving = static_cast<double>(DUCT_DRIVING_FORCE);
const double centerlineExact = driving * radiusSquared / (4.0 * eta0);
const double polymerForceExact = -etaP * driving / eta0;

for (int pz = 0; pz < NZ_TOTAL; ++pz) {
    for (int py = 0; py < NY; ++py) {
        for (int px = 0; px < NX; ++px) {
            const double dx = static_cast<double>(px) - DUCT_CENTER_X;
            const double dy = static_cast<double>(py) - DUCT_CENTER_Y;
            const double rSquared = dx * dx + dy * dy;
            if (rSquared >= radiusSquared) continue;

            const double r = sqrt(rSquared);
            const bool bulk = r < radius - 2.0;
            const int secondaryRegion = bulk ? 1 : 2;
            const double ux = ductMoment(px, py, pz, M_UX_INDEX) / F_M_I_SCALE;
            const double uy = ductMoment(px, py, pz, M_UY_INDEX) / F_M_I_SCALE;
            const double uz = ductUz(px, py, pz);
            const double ur = r > 0.0 ? (dx * ux + dy * uy) / r : 0.0;
            const double uzExact = driving * (radiusSquared - rSquared) /
                (4.0 * eta0);
            const double duzdxExact = -driving * dx / (2.0 * eta0);
            const double duzdyExact = -driving * dy / (2.0 * eta0);
            const double shearExact = -driving * r / (2.0 * eta0);

            double numericalA[6];
            for (int component = 0; component < 6; ++component)
                numericalA[component] = ductA(px, py, pz, aMoments[component]);
            const double exactA[6] = {
                1.0, 0.0, lambda * duzdxExact,
                1.0, lambda * duzdyExact,
                1.0 + 2.0 * lambda * lambda *
                    (duzdxExact * duzdxExact + duzdyExact * duzdyExact)
            };
            const double equilibriumA[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};

            const double velocityError = uz - uzExact;
            for (int regionPass = 0; regionPass < 2; ++regionPass) {
                const int region = regionPass == 0 ? 0 : secondaryRegion;
                velocityErrorSq[region] += velocityError * velocityError;
                velocityExactSq[region] += uzExact * uzExact;
                velocityLinf[region] = fmax(velocityLinf[region], fabs(velocityError));
                for (int component = 0; component < 6; ++component) {
                    const double error = numericalA[component] - exactA[component];
                    aErrorSq[component][region] += error * error;
                    const double exactDisturbance =
                        exactA[component] - equilibriumA[component];
                    aExactSq[component][region] +=
                        exactDisturbance * exactDisturbance;
                    aLinf[component][region] = fmax(
                        aLinf[component][region], fabs(error));
                }
                ++regionCount[region];
            }

            flowRateSum += uz;
            maxRadialVelocity = fmax(maxRadialVelocity, fabs(ur));

            const double cosTheta = r > 0.0 ? dx / r : 1.0;
            const double sinTheta = r > 0.0 ? dy / r : 0.0;
            const double Arz = cosTheta * numericalA[2] + sinTheta * numericalA[4];
            const double ArzExact = lambda * shearExact;
            const double Arr = cosTheta * cosTheta * numericalA[0]
                + 2.0 * cosTheta * sinTheta * numericalA[1]
                + sinTheta * sinTheta * numericalA[3];
            const double tauPrz = etaP * Arz / lambda;
            const double n1 = etaP * (numericalA[5] - Arr) / lambda;

            const double leadingMinor2 = numericalA[0] * numericalA[3]
                - numericalA[1] * numericalA[1];
            const double determinant =
                numericalA[0] * (numericalA[3] * numericalA[5]
                    - numericalA[4] * numericalA[4])
              - numericalA[1] * (numericalA[1] * numericalA[5]
                    - numericalA[4] * numericalA[2])
              + numericalA[2] * (numericalA[1] * numericalA[4]
                    - numericalA[3] * numericalA[2]);
            minimumPrincipalMinor = fmin(minimumPrincipalMinor,
                fmin(numericalA[0], fmin(leadingMinor2, determinant)));

            if (!bulk) {
                const int xm = (px - 1 + NX) % NX;
                const int xp = (px + 1) % NX;
                const int ym = (py - 1 + NY) % NY;
                const int yp = (py + 1) % NY;
                const double duzdx = 0.5 * (ductUz(xp, py, pz) - ductUz(xm, py, pz));
                const double duzdy = 0.5 * (ductUz(px, yp, pz) - ductUz(px, ym, pz));
                const double radialShear = cosTheta * duzdx + sinTheta * duzdy;
                maxNearWallShearError = fmax(
                    maxNearWallShearError, fabs(radialShear - shearExact));
            } else {
                const int xm = (px - 1 + NX) % NX;
                const int xp = (px + 1) % NX;
                const int ym = (py - 1 + NY) % NY;
                const int yp = (py + 1) % NY;
                const double dAxzdx = 0.5 * (
                    ductA(xp, py, pz, A_XZ_C_INDEX) -
                    ductA(xm, py, pz, A_XZ_C_INDEX));
                const double dAyzdy = 0.5 * (
                    ductA(px, yp, pz, A_YZ_C_INDEX) -
                    ductA(px, ym, pz, A_YZ_C_INDEX));
                const double polymerForce = etaP * (dAxzdx + dAyzdy) / lambda;
                const double polymerForceError = polymerForce - polymerForceExact;
                polymerForceSum += polymerForce;
                polymerForceErrorSq += polymerForceError * polymerForceError;
                ++polymerForceCount;
            }

            const int bin = std::min(ductBins - 1, static_cast<int>(r));
            radiusSum[bin] += r;
            uzSum[bin] += uz;
            uzExactSum[bin] += uzExact;
            urSum[bin] += ur;
            arzSum[bin] += Arz;
            arzExactSum[bin] += ArzExact;
            azzSum[bin] += numericalA[5];
            azzExactSum[bin] += exactA[5];
            tauPrzSum[bin] += tauPrz;
            n1Sum[bin] += n1;
            ++binCount[bin];
        }
    }
}

// A fully developed periodic solution is independent of z. Report the worst
// peak-to-peak variation at a fixed cross-section node for each field.
double maxAxialUzVariation = 0.0;
double maxAxialAVariation[6] = {};
for (int py = 0; py < NY; ++py) {
    for (int px = 0; px < NX; ++px) {
        const double dx = static_cast<double>(px) - DUCT_CENTER_X;
        const double dy = static_cast<double>(py) - DUCT_CENTER_Y;
        if (dx * dx + dy * dy >= radiusSquared) continue;

        double uzMinimum = ductUz(px, py, 0);
        double uzMaximum = uzMinimum;
        double aMinimum[6], aMaximum[6];
        for (int component = 0; component < 6; ++component) {
            aMinimum[component] = ductA(px, py, 0, aMoments[component]);
            aMaximum[component] = aMinimum[component];
        }
        for (int pz = 1; pz < NZ_TOTAL; ++pz) {
            const double uz = ductUz(px, py, pz);
            uzMinimum = fmin(uzMinimum, uz);
            uzMaximum = fmax(uzMaximum, uz);
            for (int component = 0; component < 6; ++component) {
                const double value = ductA(px, py, pz, aMoments[component]);
                aMinimum[component] = fmin(aMinimum[component], value);
                aMaximum[component] = fmax(aMaximum[component], value);
            }
        }
        maxAxialUzVariation = fmax(
            maxAxialUzVariation, uzMaximum - uzMinimum);
        for (int component = 0; component < 6; ++component) {
            maxAxialAVariation[component] = fmax(
                maxAxialAVariation[component],
                aMaximum[component] - aMinimum[component]);
        }
    }
}

double centerlineNumerical = 0.0;
const int cx0 = static_cast<int>(floor(DUCT_CENTER_X));
const int cy0 = static_cast<int>(floor(DUCT_CENTER_Y));
for (int pz = 0; pz < NZ_TOTAL; ++pz) {
    centerlineNumerical += 0.25 * (
        ductUz(cx0, cy0, pz) + ductUz(cx0 + 1, cy0, pz) +
        ductUz(cx0, cy0 + 1, pz) + ductUz(cx0 + 1, cy0 + 1, pz));
}
centerlineNumerical /= static_cast<double>(NZ_TOTAL);

const double numericalQ = flowRateSum / static_cast<double>(NZ_TOTAL);
const double exactQ = M_PI * driving * radiusSquared * radiusSquared /
    (8.0 * eta0);
const double epsilon = 1.0e-30;
const char* regionNames[3] = {"full", "bulk", "wall"};
std::ostringstream summary;
summary << std::scientific << std::setprecision(12)
    << "oldroyd_b_ibm_validation," << step;
for (int region = 0; region < 3; ++region) {
    summary << "," << regionNames[region] << "_uz_l2,"
            << sqrt(velocityErrorSq[region] /
                    fmax(velocityExactSq[region], epsilon))
            << "," << regionNames[region] << "_uz_linf,"
            << velocityLinf[region];
    for (int component = 0; component < 6; ++component) {
        const double componentMetric = aExactSq[component][region] > epsilon
            ? sqrt(aErrorSq[component][region] / aExactSq[component][region])
            : sqrt(aErrorSq[component][region] /
                fmax(static_cast<double>(regionCount[region]), 1.0));
        summary << "," << regionNames[region] << "_" << aNames[component]
                << "_error," << componentMetric
                << "," << regionNames[region] << "_" << aNames[component]
                << "_linf," << aLinf[component][region];
    }
}
summary << ",centerline," << centerlineNumerical
    << ",centerline_exact," << centerlineExact
    << ",q," << numericalQ << ",q_exact," << exactQ
    << ",q_relative_error," << fabs(numericalQ - exactQ) / exactQ
    << ",max_abs_ur," << maxRadialVelocity
    << ",max_near_wall_shear_error," << maxNearWallShearError
    << ",max_axial_uz_variation," << maxAxialUzVariation;
for (int component = 0; component < 6; ++component) {
    summary << ",max_axial_" << aNames[component] << "_variation,"
            << maxAxialAVariation[component];
}
summary
    << ",minimum_principal_minor," << minimumPrincipalMinor
    << ",bulk_polymer_force_mean,"
    << polymerForceSum / fmax(static_cast<double>(polymerForceCount), 1.0)
    << ",bulk_polymer_force_exact," << polymerForceExact
    << ",bulk_polymer_force_rms_error,"
    << sqrt(polymerForceErrorSq /
        fmax(static_cast<double>(polymerForceCount), 1.0));
saveTreatData("_oldroyd_b_ibm_validation", summary.str(), step);

std::ostringstream profile;
profile << std::scientific << std::setprecision(12) << step;
for (int bin = 0; bin < ductBins; ++bin) {
    if (binCount[bin] == 0) continue;
    const double inverseCount = 1.0 / static_cast<double>(binCount[bin]);
    profile << "," << radiusSum[bin] * inverseCount / radius
            << "," << uzSum[bin] * inverseCount
            << "," << uzExactSum[bin] * inverseCount
            << "," << urSum[bin] * inverseCount
            << "," << arzSum[bin] * inverseCount
            << "," << arzExactSum[bin] * inverseCount
            << "," << azzSum[bin] * inverseCount
            << "," << azzExactSum[bin] * inverseCount
            << "," << tauPrzSum[bin] * inverseCount
            << "," << n1Sum[bin] * inverseCount;
}
saveTreatData("_oldroyd_b_ibm_radial_profile", profile.str(), step);
