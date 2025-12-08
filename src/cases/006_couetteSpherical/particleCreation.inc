srand(12345);  // fixed seed for reproducibility


dfloat diameter = 20.0;
dfloat radius   = diameter / 2.0;

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


/*

srand(12345);  // fixed seed for reproducibility


dfloat diameter = 20.0;
dfloat radius   = diameter / 2.0;

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