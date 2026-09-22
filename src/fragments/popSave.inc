#if defined(STREAMING_AA_ESOTERIC_TWIST)
#include "popSave_esoteric_twist.inc"
#elif defined(STREAMING_AA_ESOTERIC_PUSH)
#include "popSave_esoteric_push.inc"
#elif defined(STREAMING_AA_ESOTERIC_PULL)
#include "popSave_esoteric_pull.inc"
#elif defined(STREAMING_AA_STD) || defined(STREAMING_AB)
#include "popSave_standard.inc"
#else
#error "No population-save streaming implementation selected"
#endif
