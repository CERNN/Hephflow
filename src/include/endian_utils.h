/**
 *  @file endian_utils.h
 *  @brief Endian conversion utilities
 *  @version 0.4.0
 *  @date 27/12/2025
 */

#ifndef __ENDIAN_UTILS_H
#define __ENDIAN_UTILS_H

#include <stdint.h>
#include <cstdint>

/* ========================== ENDIAN SWAP FUNCTIONS ======================== */

// Swap 32-bit word
static inline uint32_t swap32(uint32_t v) {
    return  (v<<24) | 
           ((v<<8)&0x00FF0000) |
           ((v>>8)&0x0000FF00) |
            (v>>24);
}

// Swap 64-bit word
static inline uint64_t swap64(uint64_t v) {
    return  (v<<56) |
           ((v<<40)&0x00FF000000000000ULL) |
           ((v<<24)&0x0000FF0000000000ULL) |
           ((v<<8 )&0x000000FF00000000ULL) |
           ((v>>8 )&0x00000000FF000000ULL) |
           ((v>>24)&0x0000000000FF0000ULL) |
           ((v>>40)&0x000000000000FF00ULL) |
            (v>>56);
}

#endif //__ENDIAN_UTILS_H