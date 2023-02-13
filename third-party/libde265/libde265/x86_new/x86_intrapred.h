#ifndef X86_INTRAPRED_H
#define X86_INTRAPRED_H

#include <stddef.h>
#include <stdint.h>

#include "slice.h"


#if HAVE_SSE4_1
void intra_prediction_DC_4_8_sse4(uint8_t *dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border);
void intra_prediction_DC_8_8_sse4(uint8_t *dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border);
void intra_prediction_DC_16_8_sse4(uint8_t *dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border);
void intra_prediction_DC_32_8_sse4(uint8_t *dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border);

void intra_prediction_angular_27_34_sse4(uint8_t* dst, int dstStride,
                                         int bit_depth, bool disableIntraBoundaryFilter,
                                         int xB0,int yB0,
                                         enum IntraPredMode intraPredMode,
                                         int nT,int cIdx,
                                         uint8_t * border);

void intra_prediction_angular_18_26_sse4(uint8_t* dst, int dstStride,
                                         int bit_depth, bool disableIntraBoundaryFilter,
                                         int xB0,int yB0,
                                         IntraPredMode intraPredMode,
                                         int nT,int cIdx,
                                         uint8_t * border);

void intra_prediction_angular_10_17_sse4(uint8_t* dst, int dstStride,
                                         int bit_depth, bool disableIntraBoundaryFilter,
                                         int xB0,int yB0,
                                         enum IntraPredMode intraPredMode,
                                         int nT,int cIdx,
                                         uint8_t * border);

void intra_prediction_angular_2_9_sse4(uint8_t* dst, int dstStride,
                                       int bit_depth, bool disableIntraBoundaryFilter,
                                       int xB0,int yB0,
                                       enum IntraPredMode intraPredMode,
                                       int nT,int cIdx,
                                       uint8_t * border);

void intra_prediction_planar_8_sse4(uint8_t *_dst, int _dstStride, int nT,int cIdx, uint8_t *border);

void intra_prediction_sample_filtering_sse4(const seq_parameter_set& sps,
                                            uint8_t* p,
                                            int nT, int cIdx,
                                            enum IntraPredMode intraPredMode);
#endif


#if HAVE_AVX2
void intra_prediction_DC_32_8_avx2(uint8_t *dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border);
void intra_prediction_planar_8_avx2(uint8_t *_dst, int _dstStride, int nT,int cIdx, uint8_t *border);
#endif

#endif