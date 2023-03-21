#ifndef X86_IDCT_H
#define X86_IDCT_H

#include <stddef.h>
#include <stdint.h>

#if HAVE_SSE4_1
void ff_hevc_transform_4x4_luma_add_8_sse4(uint8_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth);
void ff_hevc_transform_4x4_add_8_sse4(uint8_t *dst, /*const*/int16_t *coeffs, ptrdiff_t stride, int16_t col_limit);
void ff_hevc_transform_8x8_add_8_sse4(uint8_t *dst, /*const*/int16_t *coeffs, ptrdiff_t stride, int16_t col_limit);
void ff_hevc_transform_16x16_add_8_sse4(uint8_t *dst, /*const*/int16_t *coeffs, ptrdiff_t stride, int16_t col_limit);
void ff_hevc_transform_32x32_add_8_sse4(uint8_t *dst, /*const*/int16_t *coeffs, ptrdiff_t stride, int16_t col_limit);

void ff_hevc_transform_4x4_dc_add_8_sse4(uint8_t *_dst, /*const*/ int16_t *coeffs, ptrdiff_t _stride, int16_t col_limit);
void ff_hevc_transform_8x8_dc_add_8_sse4(uint8_t *_dst, /*const*/ int16_t *coeffs, ptrdiff_t _stride, int16_t col_limit);
void ff_hevc_transform_16x16_dc_add_8_sse4(uint8_t *_dst, /*const*/ int16_t *coeffs, ptrdiff_t _stride, int16_t col_limit);
void ff_hevc_transform_32x32_dc_add_8_sse4(uint8_t *_dst, /*const*/ int16_t *coeffs, ptrdiff_t _stride, int16_t col_limit);

void ff_hevc_residual16_add_8_sse4(uint8_t* _dst, ptrdiff_t _stride, const int16_t* coeffs, int nT, int bit_depth);
void ff_hevc_transform_skip_residual16_sse4 (int16_t *residual, const int16_t *coeffs, int nT, int tsShift,int bdShift);
#endif

#if HAVE_AVX2
void ff_hevc_transform_16x16_add_8_avx2(uint8_t *dst, /*const*/int16_t *coeffs, ptrdiff_t stride, int16_t col_limit);
void ff_hevc_transform_32x32_add_8_avx2(uint8_t *dst, /*const*/int16_t *coeffs, ptrdiff_t stride, int16_t col_limit);
void ff_hevc_transform_16x16_dc_add_8_avx2(uint8_t *_dst, /*const*/ int16_t *coeffs, ptrdiff_t _stride, int16_t col_limit);
void ff_hevc_transform_32x32_dc_add_8_avx2(uint8_t *_dst, /*const*/ int16_t *coeffs, ptrdiff_t _stride, int16_t col_limit);
void ff_hevc_transform_skip_residual16_avx2 (int16_t *residual, const int16_t *coeffs, int nT, int tsShift,int bdShift);
#endif

#endif