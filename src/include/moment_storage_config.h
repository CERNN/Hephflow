/**
 * @file moment_storage_config.h
 * @brief Compile-time FP16 storage switch for second-order moments.
 *
 * Arithmetic remains in FP32. Density and velocity always remain in FP32;
 * only the six second-order moments may be stored as scaled FP16.
 */
#ifndef HEPHFLOW_MOMENT_STORAGE_CONFIG_H
#define HEPHFLOW_MOMENT_STORAGE_CONFIG_H

#ifndef USE_FP16_SECOND_ORDER_MOMENTS
#define USE_FP16_SECOND_ORDER_MOMENTS 0
#endif

#if USE_FP16_SECOND_ORDER_MOMENTS != 0 && USE_FP16_SECOND_ORDER_MOMENTS != 1
#error "USE_FP16_SECOND_ORDER_MOMENTS must be 0 or 1"
#endif

#ifndef MOM_SCALE_MII
#define MOM_SCALE_MII 1048576.0f
#endif
#ifndef MOM_SCALE_MIJ
#define MOM_SCALE_MIJ 1048576.0f
#endif

__host__ __device__ constexpr bool momentStorageUsesHalf(int moment)
{
#if USE_FP16_SECOND_ORDER_MOMENTS
    return moment >= M_MXX_INDEX && moment <= M_MZZ_INDEX;
#else
    (void)moment;
    return false;
#endif
}

__host__ __device__ constexpr size_t momentStorageElementBytes(int moment)
{
    return momentStorageUsesHalf(moment) ? sizeof(unsigned short) : sizeof(dfloat);
}

constexpr size_t momentStorageBytesPerNode()
{
    size_t bytes = 0;
    for (int moment = 0; moment < static_cast<int>(NUMBER_MOMENTS); ++moment)
        bytes += momentStorageElementBytes(moment);
    return bytes;
}

__host__ __device__ constexpr dfloat momentStorageScale(int moment)
{
#if USE_FP16_SECOND_ORDER_MOMENTS
    if (moment == M_MXX_INDEX || moment == M_MYY_INDEX || moment == M_MZZ_INDEX)
        return static_cast<dfloat>(MOM_SCALE_MII);
    if (moment == M_MXY_INDEX || moment == M_MXZ_INDEX || moment == M_MYZ_INDEX)
        return static_cast<dfloat>(MOM_SCALE_MIJ);
#else
    (void)moment;
#endif
    return 1.0_df;
}

constexpr size_t MOM_STORAGE_BYTES_PER_NODE = momentStorageBytesPerNode();

#if USE_FP16_SECOND_ORDER_MOMENTS
static_assert(NUMBER_MOMENTS == 10,
    "FP16 moment storage supports only the ten-moment Newtonian solver");
static_assert(sizeof(dfloat) == sizeof(float),
    "FP16 moment storage requires FP32 dfloat arithmetic");
static_assert(N_GPUS == 1,
    "FP16 moment storage is currently restricted to one GPU");
#if LOAD_CHECKPOINT || CHECKPOINT_SAVE || MEAN_FLOW
#error "FP16 moment storage does not support checkpoint or mean-flow buffers"
#endif
static_assert(MOM_SCALE_MII > 0.0f && MOM_SCALE_MIJ > 0.0f,
    "Moment storage scales must be positive");
#endif

#endif
