#ifndef X86_SAO_H
#define X86_SAO_H

#include <stddef.h>
#include <stdint.h>

#if HAVE_SSE4_1
void ff_hevc_sao_band_filter_8_sse4(uint8_t *_dst, int stride_dst, /*const*/ uint8_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, 
                                    int saoLeftClass, const int8_t *saoOffsetVal, int bandShift, const int maxPixelValue);
void ff_hevc_sao_band_filter_16_sse4(uint16_t *_dst, int stride_dst, /*const*/ uint16_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, 
                           int saoLeftClass, const int8_t *saoOffsetVal, int bandshift, const int maxPixelValue) ;
void ff_hevc_sao_edge_filter_8_sse4(uint8_t* out_ptr, int out_stride, uint8_t* in_ptr , int in_stride, int SaoEoClass, int8_t* saoOffsetVal, 
                                    int* edges, int ctuW, int ctuH, const int maxPixelValue);
#endif

#if HAVE_AVX2
void ff_hevc_sao_band_filter_8_avx2(uint8_t *_dst, int stride_dst, /*const*/ uint8_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, 
                                    int saoLeftClass, const int8_t *saoOffsetVal, int bandShift, const int maxPixelValue);
void ff_hevc_sao_band_filter_16_avx2(uint16_t *_dst, int stride_dst, /*const*/ uint16_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, 
                           int saoLeftClass, const int8_t *saoOffsetVal, int bandshift, const int maxPixelValue) ;
void ff_hevc_sao_edge_filter_8_avx2(uint8_t* out_ptr, int out_stride, uint8_t* in_ptr , int in_stride, int SaoEoClass, int8_t* saoOffsetVal, 
                                    int* edges, int ctuW, int ctuH, const int maxPixelValue);
#endif

#endif