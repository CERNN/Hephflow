// Separate successive users of the shared population tile, then perform the
// same in-block scalar pull as the production collision kernel.
activeBarrier.arrive_and_wait();
#include "convection_diffusion_streaming.inc"
