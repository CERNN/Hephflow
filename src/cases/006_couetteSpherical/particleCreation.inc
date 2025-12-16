srand(12345);  // fixed seed for reproducibility


dfloat diameter = 16.0;
dfloat radius   = diameter / 2.0;

srand(12345);  // fixed seed for reproducibility

/*
int MAX_CANDIDATES = 30;

for (int i = 0; i < NUM_PARTICLES; i++) {
    ParticleCenter& center = centerStorage[i];
    dfloat3 pos;
    bool valid = false;

    while (!valid) {
        int parent;
        if (i == 0) {
            pos.x = (dfloat)((dfloat)rand() / RAND_MAX) * (NX - 1 - 2*radius) + radius;
            pos.y = (dfloat)((dfloat)rand() / RAND_MAX) * (NY - 1 - 2*radius) + radius;
            pos.z = (dfloat)((dfloat)rand() / RAND_MAX) * (NZ - 1 - 2*radius) + radius;
            valid = true;
            break;
        }

        parent = (parent+1) % i;   
        dfloat3 base = centerStorage[parent].getPos();

        for (int attempt = 0; attempt < MAX_CANDIDATES; attempt++) {
            dfloat r = diameter + ((dfloat)rand() / RAND_MAX) * diameter/2; //<<< change factor to make it close or higher
            dfloat theta = ((dfloat)rand() / RAND_MAX) * 2.0 * M_PI;
            dfloat phi   = acos(2.0*((dfloat)rand()/RAND_MAX)-1.0);

            dfloat dx = r * sin(phi) * cos(theta);
            dfloat dy = r * sin(phi) * sin(theta);
            dfloat dz = r * cos(phi);

            pos.x = base.x + dx;
            pos.y = base.y + dy;
            pos.z = base.z + dz;

            if (pos.x < radius || pos.x > (NX-1-radius)) continue;
            if (pos.y < radius || pos.y > (NY-1-radius)) continue;
            if (pos.z < radius || pos.z > (NZ-1-radius)) continue;

            bool ok = true;

            for (int j = 0; j < i; j++) {
                dfloat3 other = centerStorage[j].getPos();

                dfloat ddx = pos.x - other.x;
                dfloat ddy = pos.y - other.y;
                dfloat ddz = pos.z - other.z;
                dfloat dist = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);

                if (dist < diameter) {
                    ok = false;
                    break;
                }
            }

            if (ok) {
                valid = true;
                break;
            }
        }
    }
    printf("Particle index %d Pos %f %f %f\n", i, pos.x, pos.y, pos.z);

    dfloat3 vel = {0.0, 0.0, 0.0};
    dfloat3 sa  = {10, 20, 30};

    this->pShape[i] = SPHERE;
    center.setPos(pos);
    center.setPos_old(pos);
    center.setVel(vel);
    center.setVel_old(vel);
    center.setDensity(1.166666);
    center.setDiameter(diameter);
    center.setSemiAxis1(sa);
    center.setMovable(true);
    center.setQ_pos(dfloat4(1,0,0,0));
        
    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);
}
*/

/*

srand(12345);  // fixed seed for reproducibility

for (int i = 0; i < NUM_PARTICLES; i++) {
    ParticleCenter& center = centerStorage[i];
    dfloat3 pos;
    bool valid = false;

    while (!valid) {
        pos.x = (dfloat)((dfloat)rand() / RAND_MAX) * (NX - 1 - 2*radius) + radius;
        pos.y = (dfloat)((dfloat)rand() / RAND_MAX) * (NY - 1 - 2*radius) + radius;
        pos.z = (dfloat)((dfloat)rand() / RAND_MAX) * (NZ - 1 - 2*radius) + radius;
        
        valid = true;

        for (int j = 0; j < i; j++) {
            dfloat3 otherPos = centerStorage[j].getPos();
            dfloat dx = pos.x - otherPos.x;
            dfloat dy = pos.y - otherPos.y;
            dfloat dz = pos.z - otherPos.z;
            dfloat dist = sqrt(dx*dx + dy*dy + dz*dz);

            if (dist < diameter) {
                valid = false;
                break;
            }
        }
    }
    printf("Particle index %d Pos %f %f %f \n",i,pos.x,pos.y,pos.z);


    dfloat3 vel = {0.0, 0.0, 0.0};
    dfloat3 sa  = {10, 20, 30};

    this->pShape[i] = SPHERE;
    center.setPos(pos);
    center.setPos_old(pos);
    center.setVel(vel);
    center.setVel_old(vel);
    center.setDensity(1.166666);
    center.setDiameter(diameter);
    center.setSemiAxis1(sa);
    center.setMovable(true);
    center.setQ_pos(dfloat4(1,0,0,0));
        
    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);
}
*/


dfloat noise_amp = 0.02 * diameter;   // optional, small

// effective domain (center admissible region)
dfloat Lx = (NX - 1) - 2.0 * radius;
dfloat Ly = (NY - 1) - 2.0 * radius;
dfloat Lz = (NZ - 1) - 2.0 * radius;

std::mt19937 rng(12345);
std::uniform_real_distribution<dfloat> urand(-1.0, 1.0);

int Nc = (int)ceil((dfloat)NUM_PARTICLES / 4.0);   // FCC: 4 per cell

dfloat scale = cbrt((dfloat)Nc / (Lx * Ly * Lz));

int nx = (int)round(Lx * scale); if (nx < 1) nx = 1;
int ny = (int)round(Ly * scale); if (ny < 1) ny = 1;
int nz = (int)round(Lz * scale); if (nz < 1) nz = 1;

// ensure enough cells
while (nx * ny * nz < Nc) {
    if (nx <= ny && nx <= nz) nx++;
    else if (ny <= nz) ny++;
    else nz++;
}


bool valid = false;

while (!valid) {

    dfloat ax = Lx / nx;
    dfloat ay = Ly / ny;
    dfloat az = Lz / nz;

    dfloat dxy = 0.5 * sqrt(ax*ax + ay*ay);
    dfloat dxz = 0.5 * sqrt(ax*ax + az*az);
    dfloat dyz = 0.5 * sqrt(ay*ay + az*az);

    dfloat dmin = dxy;
    if (dxz < dmin) dmin = dxz;
    if (dyz < dmin) dmin = dyz;

    if (dmin >= diameter) {
        valid = true;
    } else {
        // reduce density in tightest direction
        if (dxy <= dxz && dxy <= dyz) {
            if (ax < ay) nx--;
            else         ny--;
        } else if (dxz <= dyz) {
            if (ax < az) nx--;
            else         nz--;
        } else {
            if (ay < az) ny--;
            else         nz--;
        }

        if (nx < 1) nx = 1;
        if (ny < 1) ny = 1;
        if (nz < 1) nz = 1;
    }
}

// final spacings
dfloat ax = Lx / nx;
dfloat ay = Ly / ny;
dfloat az = Lz / nz;

int maxSites = nx * ny * nz * 4;
dfloat *sx = new dfloat[maxSites];
dfloat *sy = new dfloat[maxSites];
dfloat *sz = new dfloat[maxSites];

int siteCount = 0;

// FCC basis
dfloat bx[4] = {0.0, 0.5, 0.5, 0.0};
dfloat by[4] = {0.0, 0.5, 0.0, 0.5};
dfloat bz[4] = {0.0, 0.0, 0.5, 0.5};

for (int iz = 0; iz < nz; iz++) {
for (int iy = 0; iy < ny; iy++) {
for (int ix = 0; ix < nx; ix++) {

    dfloat x0 = radius + ix * ax;
    dfloat y0 = radius + iy * ay;
    dfloat z0 = radius + iz * az;

    for (int b = 0; b < 4; b++) {

        dfloat x = x0 + bx[b] * ax;
        dfloat y = y0 + by[b] * ay;
        dfloat z = z0 + bz[b] * az;

        if (x < radius || x > NX - 1 - radius) continue;
        if (y < radius || y > NY - 1 - radius) continue;
        if (z < radius || z > NZ - 1 - radius) continue;

        sx[siteCount] = x;
        sy[siteCount] = y;
        sz[siteCount] = z;
        siteCount++;
    }
}}}

if (siteCount < NUM_PARTICLES) {
    printf("ERROR: FCC sites (%d) < NUM_PARTICLES (%d)\n",
           siteCount, NUM_PARTICLES);
    exit(1);
}

// Shuffle sites


for (int i = siteCount - 1; i > 0; i--) {
    std::uniform_int_distribution<int> irand(0, i);
    int j = irand(rng);

    dfloat tx = sx[i]; sx[i] = sx[j]; sx[j] = tx;
    dfloat ty = sy[i]; sy[i] = sy[j]; sy[j] = ty;
    dfloat tz = sz[i]; sz[i] = sz[j]; sz[j] = tz;
}

// Assign particles

for (int i = 0; i < NUM_PARTICLES; i++) {

    dfloat3 pos;
    pos.x = sx[i];
    pos.y = sy[i];
    pos.z = sz[i];

    // optional noise
    pos.x += noise_amp * urand(rng);
    pos.y += noise_amp * urand(rng);
    pos.z += noise_amp * urand(rng);

    centerStorage[i].setPos(pos);
    centerStorage[i].setPos_old(pos);
}

//rescale 

dfloat xmin = 1e30, xmax = -1e30;
dfloat ymin = 1e30, ymax = -1e30;
dfloat zmin = 1e30, zmax = -1e30;

for (int i = 0; i < NUM_PARTICLES; i++) {
    dfloat3 x = centerStorage[i].getPos();

    if (x.x < xmin) xmin = x.x;
    if (x.x > xmax) xmax = x.x;
    if (x.y < ymin) ymin = x.y;
    if (x.y > ymax) ymax = x.y;
    if (x.z < zmin) zmin = x.z;
    if (x.z > zmax) zmax = x.z;
}

// target admissible bounds
dfloat tx0 = radius;
dfloat tx1 = NX - 1 - radius;
dfloat ty0 = radius;
dfloat ty1 = NY - 1 - radius;
dfloat tz0 = radius;
dfloat tz1 = NZ - 1 - radius;

// scaling factors
dfloat sxg = (tx1 - tx0) / (xmax - xmin);
dfloat syg = (ty1 - ty0) / (ymax - ymin);
dfloat szg = (tz1 - tz0) / (zmax - zmin);

// rescale all particles
for (int i = 0; i < NUM_PARTICLES; i++) {

    dfloat3 x = centerStorage[i].getPos();

    x.x = tx0 + (x.x - xmin) * sxg;
    x.y = ty0 + (x.y - ymin) * syg;
    x.z = tz0 + (x.z - zmin) * szg;

    centerStorage[i].setPos(x);
    centerStorage[i].setPos_old(x);
}


for (int i = 0; i < NUM_PARTICLES; i++) {

    ParticleCenter& center = centerStorage[i];

    dfloat3 vel = {0.0, 0.0, 0.0};
    dfloat3 sa  = {10, 20, 30};

    this->pShape[i] = SPHERE;
    center.setVel(vel);
    center.setVel_old(vel);
    center.setDensity(1.166666);
    center.setDiameter(diameter);
    center.setSemiAxis1(sa);
    center.setMovable(true);
    center.setQ_pos(dfloat4(1,0,0,0));

    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);
}

// cleanup
delete[] sx;
delete[] sy;
delete[] sz;


/*
dfloat noise_amp = 0.02 * diameter;   // <= 2-3% is safe here
// effective domain size
dfloat minx = radius;
dfloat miny = radius;
dfloat minz = radius;
dfloat maxx = NX - 1.0 - radius;
dfloat maxy = NY - 1.0 - radius;
dfloat maxz = NZ - 1.0 - radius;
dfloat Lx = maxx - minx;
dfloat Ly = maxy - miny;
dfloat Lz = maxz - minz;
// RNG
std::mt19937 rng(12345);
std::uniform_real_distribution<dfloat> urand(-1.0, 1.0);

dfloat low = diameter;
dfloat high = std::max({Lx, Ly, Lz}) + 1.0;
dfloat eps = 1e-9f;
long long target = (long long)NUM_PARTICLES;
for (int iter = 0; iter < 100; iter++) {
    dfloat mid = (low + high) / 2.0;
    int px = (int)std::floor(Lx / mid) + 1;
    int py = (int)std::floor(Ly / mid) + 1;
    int pz = (int)std::floor(Lz / mid) + 1;
    long long pp = (long long)px * py * pz;
    if (pp >= target) {
        low = mid;
    } else {
        high = mid;
    }
}
dfloat opt_s = low;


int nx = (int)std::floor(Lx / opt_s) + 1;
int ny = (int)std::floor(Ly / opt_s) + 1;
int nz = (int)std::floor(Lz / opt_s) + 1;
long long pp = (long long)nx * ny * nz;
// Adjust if floating-point precision caused undercount
if (pp < target) {
    opt_s *= 0.999f;
    nx = (int)std::floor(Lx / opt_s) + 1;
    ny = (int)std::floor(Ly / opt_s) + 1;
    nz = (int)std::floor(Lz / opt_s) + 1;
    pp = (long long)nx * ny * nz;
}
// Verify we can fit (max possible at min spacing)
dfloat min_s = diameter;
int max_nx = (int)std::floor(Lx / min_s) + 1;
int max_ny = (int)std::floor(Ly / min_s) + 1;
int max_nz = (int)std::floor(Lz / min_s) + 1;
long long max_pp = (long long)max_nx * max_ny * max_nz;
if (max_pp < target) {
    printf("ERROR: Cannot fit %lld particles, max possible is %lld\n", target, max_pp);
    exit(1);
}
if (pp < target) {
    printf("ERROR: Site count (%lld) < NUM_PARTICLES (%lld)\n", pp, target);
    exit(1);
}
// ------------------ Force overfill if perfect fit -------------------

// Try to increase each direction if possible without spacing < diameter
if (nx >= 1) {
    dfloat test_ax = (nx + 1 > 1) ? Lx / nx : 0.0;
    if (test_ax >= diameter) {
        nx++;
    }
}
if (ny >= 1) {
    dfloat test_ay = (ny + 1 > 1) ? Ly / ny : 0.0;
    if (test_ay >= diameter) {
        ny++;
    }
}
if (nz >= 1) {
    dfloat test_az = (nz + 1 > 1) ? Lz / nz : 0.0;
    if (test_az >= diameter) {
        nz++;
    }
}
pp = (long long)nx * ny * nz;

// ------------------ Compute actual spacings -------------------------
dfloat ax = (nx > 1) ? Lx / (nx - 1) : 0.0;
dfloat ay = (ny > 1) ? Ly / (ny - 1) : 0.0;
dfloat az = (nz > 1) ? Lz / (nz - 1) : 0.0;
// ------------------ Generate sites ----------------------------------
int maxSites = (int)pp;
dfloat *sx = new dfloat[maxSites];
dfloat *sy = new dfloat[maxSites];
dfloat *sz = new dfloat[maxSites];
int siteCount = 0;
dfloat x_mid = minx + Lx * 0.5;
dfloat y_mid = miny + Ly * 0.5;
dfloat z_mid = minz + Lz * 0.5;
for (int iz = 0; iz < nz; iz++) {
    dfloat z = (nz > 1) ? minz + (dfloat)iz * az : z_mid;
    z += noise_amp * urand(rng);
    for (int iy = 0; iy < ny; iy++) {
        dfloat y = (ny > 1) ? miny + (dfloat)iy * ay : y_mid;
        y += noise_amp * urand(rng);
        for (int ix = 0; ix < nx; ix++) {
            dfloat x = (nx > 1) ? minx + (dfloat)ix * ax : x_mid;
            x += noise_amp * urand(rng);
            sx[siteCount] = x;
            sy[siteCount] = y;
            sz[siteCount] = z;
            siteCount++;
        }
    }
}
// ------------------ Fisher Yates shuffle ----------------------------
for (int i = siteCount - 1; i > 0; i--) {
    std::uniform_int_distribution<int> irand(0, i);
    int j = irand(rng);
    dfloat tx = sx[i]; sx[i] = sx[j]; sx[j] = tx;
    dfloat ty = sy[i]; sy[i] = sy[j]; sy[j] = ty;
    dfloat tz = sz[i]; sz[i] = sz[j]; sz[j] = tz;
}

for (int i = 0; i < NUM_PARTICLES; i++) {
    dfloat3 pos;
    pos.x = sx[i] + noise_amp * urand(rng);
    pos.y = sy[i] + noise_amp * urand(rng);
    pos.z = sz[i] + noise_amp * urand(rng);
    // clamp
    if (pos.x < radius) pos.x = radius;
    if (pos.x > NX - 1 - radius) pos.x = NX - 1 - radius;
    if (pos.y < radius) pos.y = radius;
    if (pos.y > NY - 1 - radius) pos.y = NY - 1 - radius;
    if (pos.z < radius) pos.z = radius;
    if (pos.z > NZ - 1 - radius) pos.z = NZ - 1 - radius;
    ParticleCenter& center = centerStorage[i];
    dfloat3 vel = {0.0, 0.0, 0.0};
    dfloat3 sa  = {10, 20, 30};
    printf("Particle index %d Pos %f %f %f \n",i,pos.x,pos.y,pos.z);
    this->pShape[i] = SPHERE;
    center.setPos(pos);
    center.setPos_old(pos);
    center.setVel(vel);
    center.setVel_old(vel);
    center.setDensity(1.166666);
    center.setDiameter(diameter);
    center.setSemiAxis1(sa);
    center.setMovable(true);
    center.setQ_pos(dfloat4(1,0,0,0));
    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);
}
// cleanup
delete[] sx;
delete[] sy;
delete[] sz;

*/