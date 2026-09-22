#if defined(STREAMING_AA_ESOTERIC_TWIST)
#include "popLoad_esoteric_twist.inc"
#elif defined(STREAMING_AA_ESOTERIC_PUSH)
#include "popLoad_esoteric_push.inc"
#elif defined(STREAMING_AA_ESOTERIC_PULL)
#include "popLoad_esoteric_pull.inc"
#elif defined(STREAMING_AA_STD) || defined(STREAMING_AB)
#include "popLoad_standard.inc"
#else
#error "No population-load streaming implementation selected"
#endif
