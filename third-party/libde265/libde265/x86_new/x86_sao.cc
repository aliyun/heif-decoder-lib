#include "x86_sao.h"
#include "libde265/util.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <emmintrin.h> // SSE2
#include <tmmintrin.h> // SSSE3

#if HAVE_SSE4_1
#include <smmintrin.h> // SSE4.1
#endif

#if HAVE_AVX2 || HAVE_AVX512
#include <immintrin.h> //AVX
#endif

#include <string.h>

#if HAVE_SSE4_1
void ff_hevc_sao_band_filter_8_sse4(uint8_t *_dst, int stride_dst, /*const*/ uint8_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, int saoLeftClass, const int8_t *saoOffsetVal, int bandShift, const int maxPixelValue)
{
    uint8_t* src = _src;
    uint8_t* dst = _dst;

    __m128i r0, r1, r2, r3;
    __m128i x0, x1, x2, x3;
    __m128i sao1, sao2, sao3, sao4;
    __m128i src8, src16, src2;

    int shift = 8-5; //bitdepth -5


    r0   = _mm_set1_epi16((saoLeftClass    ) & 31);
    r1   = _mm_set1_epi16((saoLeftClass + 1) & 31);
    r2   = _mm_set1_epi16((saoLeftClass + 2) & 31);
    r3   = _mm_set1_epi16((saoLeftClass + 3) & 31);
    sao1 = _mm_set1_epi16(saoOffsetVal[0]);        
    sao2 = _mm_set1_epi16(saoOffsetVal[1]);        
    sao3 = _mm_set1_epi16(saoOffsetVal[2]);        
    sao4 = _mm_set1_epi16(saoOffsetVal[3]);

    for (int y = 0; y < ctuH; y++)
    {
        for (int x = 0; x < ctuW; x+=8)
        {
            src8 = _mm_loadl_epi64((__m128i *) (src+x)); 
            src16 = _mm_unpacklo_epi8(src8, _mm_setzero_si128());
            src2 = _mm_srai_epi16(src16, shift);  

            x0   = _mm_cmpeq_epi16(src2, r0); 
            x1   = _mm_cmpeq_epi16(src2, r1); 
            x2   = _mm_cmpeq_epi16(src2, r2); 
            x3   = _mm_cmpeq_epi16(src2, r3);

            x0   = _mm_and_si128(x0, sao1);
            x1   = _mm_and_si128(x1, sao2);
            x2   = _mm_and_si128(x2, sao3);
            x3   = _mm_and_si128(x3, sao4);

            x0   = _mm_or_si128(x0, x1);
            x2   = _mm_or_si128(x2, x3);
            x0   = _mm_or_si128(x0, x2);

            src16 = _mm_add_epi16(src16, x0);
            src8 = _mm_packus_epi16(src16, src16);    
            _mm_storel_epi64((__m128i *) &dst[x], src8);
        }

        src += stride_src;
        dst += stride_dst;
    }
}

void ff_hevc_sao_edge_filter_8_sse4(uint8_t* out_ptr, int out_stride, uint8_t* in_ptr , int in_stride, int SaoEoClass, int8_t* saoOffsetVal, 
                                    int* edges, int ctuW, int ctuH, const int maxPixelValue)
{
  const int8_t pos[4][2][2] = {{ {-1, 0}, { 1, 0} },{ { 0,-1}, { 0, 1} }, { {-1,-1}, { 1, 1} }, { { 1,-1}, {-1, 1} }, };
  int init_y = 0, width = ctuW, height = ctuH;
  uint8_t* dst = out_ptr;
  uint8_t* src = in_ptr;
  int stride_dst = out_stride, stride_src = in_stride;

  if (SaoEoClass != 0) {
    __m128i x0, x1;
    __m128i src8;
    if (edges[1]) {
      x1 = _mm_set1_epi16(saoOffsetVal[2]);
      for (int x = 0; x < width; x += 8) {
        src8 = _mm_loadl_epi64((__m128i *)(src + x));
        x0  = _mm_unpacklo_epi8(src8, _mm_setzero_si128());
        x0 = _mm_add_epi16(x0, x1);
        src8 = _mm_packus_epi16(x0, x0);
         _mm_storel_epi64((__m128i *) (dst+x), src8);
      };
      init_y = 1;
    }
    if (edges[3]) {
      int y_stride_dst = stride_dst * (ctuH - 1);
      int y_stride_src = stride_src * (ctuH - 1);
      x1 = _mm_set1_epi16(saoOffsetVal[2]);
      for (int x = 0; x < width; x += 8) {
        src8 = _mm_loadl_epi64((__m128i *)(src + x + y_stride_src));
        x0  = _mm_unpacklo_epi8(src8, _mm_setzero_si128());
        x0 = _mm_add_epi16(x0, x1);
        src8 = _mm_packus_epi16(x0, x0);
        _mm_storel_epi64((__m128i *)(dst + x + y_stride_dst), src8);
      };
      height--;
    }
  }

  {
    int y_stride_dst = init_y * stride_dst;
    int y_stride_src = init_y * stride_src;
    int pos_0_0 = pos[SaoEoClass][0][0];
    int pos_0_1 = pos[SaoEoClass][0][1];
    int pos_1_0 = pos[SaoEoClass][1][0];
    int pos_1_1 = pos[SaoEoClass][1][1];
    int y_stride_0_1 = (init_y + pos_0_1) * stride_src + pos_0_0;
    int y_stride_1_1 = (init_y + pos_1_1) * stride_src + pos_1_0;

    __m128i offset = _mm_set_epi8(0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, saoOffsetVal[4], 
                                  saoOffsetVal[3], saoOffsetVal[2], saoOffsetVal[1], saoOffsetVal[0]);

    for (int y = init_y; y < height; y++) {
      for (int x = 0; x < width; x += 8) {
        __m128i x0    = _mm_loadl_epi64((__m128i *) (src + x + y_stride_src));
        __m128i cmp0  = _mm_loadl_epi64((__m128i *) (src + x + y_stride_0_1));
        __m128i cmp1  = _mm_loadl_epi64((__m128i *) (src + x + y_stride_1_1));

        __m128i r2    = _mm_min_epu8(x0, cmp0);
        __m128i x1    = _mm_cmpeq_epi8(cmp0, r2);
        __m128i x2    = _mm_cmpeq_epi8(x0, r2);
        __m128i diff0 = _mm_sub_epi8(x2, x1);

                r2    = _mm_min_epu8(x0, cmp1);
        __m128i x3    = _mm_cmpeq_epi8(cmp1, r2);
                x2    = _mm_cmpeq_epi8(x0, r2);
        __m128i diff1 = _mm_sub_epi8(x2, x3);

                diff0 = _mm_add_epi8(diff0, diff1);
        __m128i index = _mm_add_epi8(diff0, _mm_set1_epi8(2));

        __m128i r0    = _mm_shuffle_epi8(offset, index);
                r0 = _mm_unpacklo_epi8(r0, _mm_cmplt_epi8(r0, _mm_setzero_si128()));
                x0 = _mm_unpacklo_epi8(x0, _mm_setzero_si128());
                r0 = _mm_add_epi16(r0, x0);
                r0 = _mm_packus_epi16(r0, r0);
        _mm_storel_epi64((__m128i *) (dst + x + y_stride_dst), r0);
      }
      y_stride_dst += stride_dst;
      y_stride_src += stride_src;
      y_stride_0_1 += stride_src;
      y_stride_1_1 += stride_src;
    }
  }

  if (SaoEoClass != 1) {
    if (edges[0]) {
      int idx_dst        = 0;
      int idx_src        = 0;
      int16_t offset_val = saoOffsetVal[2];
      for (int y = 0; y < height; y++) {
        dst[idx_dst] = Clip3(0, maxPixelValue, src[idx_src] + offset_val);
        idx_dst     += stride_dst;
        idx_src     += stride_src;
      }
    }
    if (edges[2]) {
      int idx_dst        = ctuW - 1;
      int idx_src        = idx_dst;
      int16_t offset_val = saoOffsetVal[2];
      for (int y = 0; y < height; y++) {
        dst[idx_dst] = Clip3(0, maxPixelValue, src[idx_src] + offset_val);
        idx_dst     += stride_dst;
        idx_src     += stride_src;
      }
    }                                                                      
  }  
}

void ff_hevc_sao_band_filter_16_sse4(uint16_t *_dst, int stride_dst, /*const*/ uint16_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, 
                           int saoLeftClass, const int8_t *saoOffsetVal, int bandshift, const int maxPixelValue) 
{
    uint16_t* src = _src;
    uint16_t* dst = _dst;

    __m128i r0, r1, r2, r3;
    __m128i x0, x1, x2, x3;
    __m128i sao1, sao2, sao3, sao4;
    __m128i src16, src2;

    int shift = bandshift; //bitdepth -5


    r0   = _mm_set1_epi16((saoLeftClass    ) & 31);
    r1   = _mm_set1_epi16((saoLeftClass + 1) & 31);
    r2   = _mm_set1_epi16((saoLeftClass + 2) & 31);
    r3   = _mm_set1_epi16((saoLeftClass + 3) & 31);
    sao1 = _mm_set1_epi16(saoOffsetVal[0]);        
    sao2 = _mm_set1_epi16(saoOffsetVal[1]);        
    sao3 = _mm_set1_epi16(saoOffsetVal[2]);        
    sao4 = _mm_set1_epi16(saoOffsetVal[3]);

    for (int y = 0; y < ctuH; y++)
    {
        for (int x = 0; x < ctuW; x+=8)
        {
            src16 = _mm_loadu_si128((__m128i *) (src+x)); 
            src2 = _mm_srai_epi16(src16, shift);  

            x0   = _mm_cmpeq_epi16(src2, r0); 
            x1   = _mm_cmpeq_epi16(src2, r1); 
            x2   = _mm_cmpeq_epi16(src2, r2); 
            x3   = _mm_cmpeq_epi16(src2, r3);

            x0   = _mm_and_si128(x0, sao1);
            x1   = _mm_and_si128(x1, sao2);
            x2   = _mm_and_si128(x2, sao3);
            x3   = _mm_and_si128(x3, sao4);

            x0   = _mm_or_si128(x0, x1);
            x2   = _mm_or_si128(x2, x3);
            x0   = _mm_or_si128(x0, x2);

            src16 = _mm_add_epi16(src16, x0);


            src16 = _mm_max_epi16(src16, _mm_setzero_si128()); 
            src16 = _mm_min_epi16(src16, _mm_set1_epi16(maxPixelValue));
            _mm_store_si128((__m128i *) &dst[x  ], src16);
        }

        src += stride_src;
        dst += stride_dst;
    } 
}

#endif


#if HAVE_AVX2
void ff_hevc_sao_band_filter_8_avx2(uint8_t *_dst, int stride_dst, /*const*/ uint8_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, int saoLeftClass, const int8_t *saoOffsetVal, int bandShift, const int maxPixelValue)
{
    uint8_t* src = _src;
    uint8_t* dst = _dst;

    __m256i r0, r1, r2, r3;
    __m256i x0, x1, x2, x3;
    __m256i sao1, sao2, sao3, sao4;
    __m256i src16, src2;

    int shift = 8-5; //bitdepth -5


    r0   = _mm256_set1_epi16((saoLeftClass    ) & 31);
    r1   = _mm256_set1_epi16((saoLeftClass + 1) & 31);
    r2   = _mm256_set1_epi16((saoLeftClass + 2) & 31);
    r3   = _mm256_set1_epi16((saoLeftClass + 3) & 31);
    sao1 = _mm256_set1_epi16(saoOffsetVal[0]);        
    sao2 = _mm256_set1_epi16(saoOffsetVal[1]);        
    sao3 = _mm256_set1_epi16(saoOffsetVal[2]);        
    sao4 = _mm256_set1_epi16(saoOffsetVal[3]);

    for (int y = 0; y < ctuH; y++)
    {
        for (int x = 0; x < ctuW; x+=16)
        {
            __m128i input, result;
            input = _mm_loadu_si128((__m128i *) (src+x)); 
            src16 = _mm256_cvtepu8_epi16(input);
            src2 = _mm256_srai_epi16(src16, shift);  

            x0   = _mm256_cmpeq_epi16(src2, r0); 
            x1   = _mm256_cmpeq_epi16(src2, r1); 
            x2   = _mm256_cmpeq_epi16(src2, r2); 
            x3   = _mm256_cmpeq_epi16(src2, r3);

            x0   = _mm256_and_si256(x0, sao1);
            x1   = _mm256_and_si256(x1, sao2);
            x2   = _mm256_and_si256(x2, sao3);
            x3   = _mm256_and_si256(x3, sao4);

            x0   = _mm256_or_si256(x0, x1);
            x2   = _mm256_or_si256(x2, x3);
            x0   = _mm256_or_si256(x0, x2);

            src16 = _mm256_add_epi16(src16, x0);

            __m128i lo_lane = _mm256_castsi256_si128(src16);
            __m128i hi_lane = _mm256_extracti128_si256(src16, 1);
            result = _mm_packus_epi16(lo_lane, hi_lane);
            _mm_storeu_si128((__m128i *)(dst+x), result);
        }

        src += stride_src;
        dst += stride_dst;
    }    
}

void ff_hevc_sao_edge_filter_8_avx2(uint8_t* out_ptr, int out_stride, uint8_t* in_ptr , int in_stride, int SaoEoClass, int8_t* saoOffsetVal, 
                                    int* edges, int ctuW, int ctuH, const int maxPixelValue)
{
  const int8_t pos[4][2][2] = {{ {-1, 0}, { 1, 0} },{ { 0,-1}, { 0, 1} }, { {-1,-1}, { 1, 1} }, { { 1,-1}, {-1, 1} }, };
  int init_y = 0, width = ctuW, height = ctuH;
  uint8_t* dst = out_ptr;
  uint8_t* src = in_ptr;
  int stride_dst = out_stride, stride_src = in_stride;

  if (SaoEoClass != 0) {
    __m256i x0, x1;
    __m128i src8;
    __m128i lo_lane, hi_lane;
    if (edges[1]) {
      x1 = _mm256_set1_epi16(saoOffsetVal[2]);
      for (int x = 0; x < width; x += 16) {
        src8 = _mm_loadu_si128((__m128i *)(src + x));
        x0  = _mm256_cvtepu8_epi16(src8);
        x0 = _mm256_add_epi16(x0, x1);
        lo_lane = _mm256_castsi256_si128(x0);
        hi_lane = _mm256_extracti128_si256(x0, 1);
        src8 = _mm_packus_epi16(lo_lane, hi_lane);
         _mm_storeu_si128((__m128i *) (dst+x), src8);
      };
      init_y = 1;
    }
    if (edges[3]) {
      int y_stride_dst = stride_dst * (ctuH - 1);
      int y_stride_src = stride_src * (ctuH - 1);
      x1 = _mm256_set1_epi16(saoOffsetVal[2]);
      for (int x = 0; x < width; x += 16) {
        src8 = _mm_loadu_si128((__m128i *)(src + x + y_stride_src));
        x0  = _mm256_cvtepu8_epi16(src8);
        x0 = _mm256_add_epi16(x0, x1);
        lo_lane = _mm256_castsi256_si128(x0);
        hi_lane = _mm256_extracti128_si256(x0, 1);
        src8 = _mm_packus_epi16(lo_lane, hi_lane);
        _mm_storeu_si128((__m128i *)(dst + x + y_stride_dst), src8);
      };
      height--;
    }
  }

  {
    int y_stride_dst = init_y * stride_dst;
    int y_stride_src = init_y * stride_src;
    int pos_0_0 = pos[SaoEoClass][0][0];
    int pos_0_1 = pos[SaoEoClass][0][1];
    int pos_1_0 = pos[SaoEoClass][1][0];
    int pos_1_1 = pos[SaoEoClass][1][1];
    int y_stride_0_1 = (init_y + pos_0_1) * stride_src + pos_0_0;
    int y_stride_1_1 = (init_y + pos_1_1) * stride_src + pos_1_0;

    __m128i offset = _mm_set_epi8(
                                //   0, 0, 0, 0,
                                //   0, 0, 0, 0,
                                //   0, 0, 0, 0,
                                //   0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, saoOffsetVal[4], 
                                  saoOffsetVal[3], saoOffsetVal[2], saoOffsetVal[1], saoOffsetVal[0]);

    for (int y = init_y; y < height; y++) {
      for (int x = 0; x < width; x += 16) {
        __m256i x0    = _mm256_loadu_si256((__m256i *) (src + x + y_stride_src));
                x0    = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(x0));

        __m256i cmp0  = _mm256_loadu_si256((__m256i *) (src + x + y_stride_0_1));
                cmp0  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(cmp0));

        __m256i cmp1  = _mm256_loadu_si256((__m256i *) (src + x + y_stride_1_1));
                cmp1  = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(cmp1));

        __m256i r2    = _mm256_min_epu16(x0, cmp0);
        __m256i x1    = _mm256_cmpeq_epi16(cmp0, r2);
        __m256i x2    = _mm256_cmpeq_epi16(x0, r2);
        __m256i diff0 = _mm256_sub_epi16(x2, x1);

                r2    = _mm256_min_epu16(x0, cmp1);
        __m256i x3    = _mm256_cmpeq_epi16(cmp1, r2);
                x2    = _mm256_cmpeq_epi16(x0, r2);
        __m256i diff1 = _mm256_sub_epi16(x2, x3);

                diff0 = _mm256_add_epi16(diff0, diff1);
        __m256i index = _mm256_add_epi16(diff0, _mm256_set1_epi16(2));

        __m128i index_lo = _mm256_castsi256_si128(index);
        __m128i index_hi = _mm256_extracti128_si256(index, 1);
        __m128i index1   = _mm_packs_epi16(index_lo, index_hi);
        __m128i r0    = _mm_shuffle_epi8(offset, index1);
        __m256i r00  = _mm256_cvtepi8_epi16(r0);

        r00 = _mm256_add_epi16(r00, x0);

        __m128i lo_lane = _mm256_castsi256_si128(r00);
        __m128i hi_lane = _mm256_extracti128_si256(r00, 1);
        __m128i  result = _mm_packus_epi16(lo_lane, hi_lane);
        _mm_storeu_si128((__m128i *) (dst + x + y_stride_dst), result);
      }
      y_stride_dst += stride_dst;
      y_stride_src += stride_src;
      y_stride_0_1 += stride_src;
      y_stride_1_1 += stride_src;
    }
  }

  if (SaoEoClass != 1) {
    if (edges[0]) {
      int idx_dst        = 0;
      int idx_src        = 0;
      int16_t offset_val = saoOffsetVal[2];
      for (int y = 0; y < height; y++) {
        dst[idx_dst] = Clip3(0, maxPixelValue, src[idx_src] + offset_val);
        idx_dst     += stride_dst;
        idx_src     += stride_src;
      }
    }
    if (edges[2]) {
      int idx_dst        = ctuW - 1;
      int idx_src        = idx_dst;
      int16_t offset_val = saoOffsetVal[2];
      for (int y = 0; y < height; y++) {
        dst[idx_dst] = Clip3(0, maxPixelValue, src[idx_src] + offset_val);
        idx_dst     += stride_dst;
        idx_src     += stride_src;
      }
    }                                                                      
  } 
}

void ff_hevc_sao_band_filter_16_avx2(uint16_t *_dst, int stride_dst, /*const*/ uint16_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, 
                           int saoLeftClass, const int8_t *saoOffsetVal, int bandshift, const int maxPixelValue)
{
    uint16_t* src = _src;
    uint16_t* dst = _dst;

    __m256i r0, r1, r2, r3;
    __m256i x0, x1, x2, x3;
    __m256i sao1, sao2, sao3, sao4;
    __m256i src16, src2;

    int shift = bandshift;


    r0   = _mm256_set1_epi16((saoLeftClass    ) & 31);
    r1   = _mm256_set1_epi16((saoLeftClass + 1) & 31);
    r2   = _mm256_set1_epi16((saoLeftClass + 2) & 31);
    r3   = _mm256_set1_epi16((saoLeftClass + 3) & 31);
    sao1 = _mm256_set1_epi16(saoOffsetVal[0]);        
    sao2 = _mm256_set1_epi16(saoOffsetVal[1]);        
    sao3 = _mm256_set1_epi16(saoOffsetVal[2]);        
    sao4 = _mm256_set1_epi16(saoOffsetVal[3]);

    for (int y = 0; y < ctuH; y++)
    {
        for (int x = 0; x < ctuW; x+=16)
        {
            src16 = _mm256_loadu_si256((__m256i *) (src+x)); 
            src2 = _mm256_srai_epi16(src16, shift);  

            x0   = _mm256_cmpeq_epi16(src2, r0); 
            x1   = _mm256_cmpeq_epi16(src2, r1); 
            x2   = _mm256_cmpeq_epi16(src2, r2); 
            x3   = _mm256_cmpeq_epi16(src2, r3);

            x0   = _mm256_and_si256(x0, sao1);
            x1   = _mm256_and_si256(x1, sao2);
            x2   = _mm256_and_si256(x2, sao3);
            x3   = _mm256_and_si256(x3, sao4);

            x0   = _mm256_or_si256(x0, x1);
            x2   = _mm256_or_si256(x2, x3);
            x0   = _mm256_or_si256(x0, x2);

            src16 = _mm256_add_epi16(src16, x0);


            src16 = _mm256_max_epi16(src16, _mm256_setzero_si256()); 
            src16 = _mm256_min_epi16(src16, _mm256_set1_epi16(maxPixelValue));
            _mm256_storeu_si256((__m256i *) &dst[x  ], src16);
        }

        src += stride_src;
        dst += stride_dst;
    }    
}

#endif