/* Small deterministic phase-field verification case. */
#define D3Q19

#if defined(PHASE_TEST_BOTH_METHODS)
    #define METHOD_CAHN_HILLIARD
    #define METHOD_ALLEN_CAHN
#elif defined(PHASE_TEST_NO_METHOD)
    /* Used by the negative compile test. */
#elif defined(PHASE_TEST_METHOD_CH)
    #define METHOD_CAHN_HILLIARD
#else
    #define METHOD_ALLEN_CAHN
#endif

#if defined(PHASE_TEST_BOTH_FORCES)
    #define PHASE_FORCE_CHEMICAL_POTENTIAL
    #define PHASE_FORCE_CSF
#elif defined(PHASE_TEST_FORCE_CSF)
    #define PHASE_FORCE_CSF
#elif defined(PHASE_TEST_FORCE_CHEMICAL)
    #define PHASE_FORCE_CHEMICAL_POTENTIAL
#endif

#define COLLISION_TYPE MR_LBM_2ND_ORDER
#define RANDOM_NUMBERS false

