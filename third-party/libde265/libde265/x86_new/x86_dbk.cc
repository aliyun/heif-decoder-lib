#include "x86_dbk.h"
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
LIBDE265_INLINE void transpose8x8(const __m128i* const in,
                                            __m128i* const out) {
  // Unpack 16 bit elements. Goes from:
  // in[0]: 00 01 02 03  04 05 06 07
  // in[1]: 10 11 12 13  14 15 16 17
  // in[2]: 20 21 22 23  24 25 26 27
  // in[3]: 30 31 32 33  34 35 36 37
  // in[4]: 40 41 42 43  44 45 46 47
  // in[5]: 50 51 52 53  54 55 56 57
  // in[6]: 60 61 62 63  64 65 66 67
  // in[7]: 70 71 72 73  74 75 76 77
  // to:
  // a0:    00 10 01 11  02 12 03 13
  // a1:    20 30 21 31  22 32 23 33
  // a2:    40 50 41 51  42 52 43 53
  // a3:    60 70 61 71  62 72 63 73
  // a4:    04 14 05 15  06 16 07 17
  // a5:    24 34 25 35  26 36 27 37
  // a6:    44 54 45 55  46 56 47 57
  // a7:    64 74 65 75  66 76 67 77
  const __m128i a0 = _mm_unpacklo_epi16(in[0], in[1]);
  const __m128i a1 = _mm_unpacklo_epi16(in[2], in[3]);
  const __m128i a2 = _mm_unpacklo_epi16(in[4], in[5]);
  const __m128i a3 = _mm_unpacklo_epi16(in[6], in[7]);
  const __m128i a4 = _mm_unpackhi_epi16(in[0], in[1]);
  const __m128i a5 = _mm_unpackhi_epi16(in[2], in[3]);
  const __m128i a6 = _mm_unpackhi_epi16(in[4], in[5]);
  const __m128i a7 = _mm_unpackhi_epi16(in[6], in[7]);

  // Unpack 32 bit elements resulting in:
  // b0: 00 10 20 30  01 11 21 31
  // b1: 40 50 60 70  41 51 61 71
  // b2: 04 14 24 34  05 15 25 35
  // b3: 44 54 64 74  45 55 65 75
  // b4: 02 12 22 32  03 13 23 33
  // b5: 42 52 62 72  43 53 63 73
  // b6: 06 16 26 36  07 17 27 37
  // b7: 46 56 66 76  47 57 67 77
  const __m128i b0 = _mm_unpacklo_epi32(a0, a1);
  const __m128i b1 = _mm_unpacklo_epi32(a2, a3);
  const __m128i b2 = _mm_unpacklo_epi32(a4, a5);
  const __m128i b3 = _mm_unpacklo_epi32(a6, a7);
  const __m128i b4 = _mm_unpackhi_epi32(a0, a1);
  const __m128i b5 = _mm_unpackhi_epi32(a2, a3);
  const __m128i b6 = _mm_unpackhi_epi32(a4, a5);
  const __m128i b7 = _mm_unpackhi_epi32(a6, a7);

  // Unpack 64 bit elements resulting in:
  // out[0]: 00 10 20 30  40 50 60 70
  // out[1]: 01 11 21 31  41 51 61 71
  // out[2]: 02 12 22 32  42 52 62 72
  // out[3]: 03 13 23 33  43 53 63 73
  // out[4]: 04 14 24 34  44 54 64 74
  // out[5]: 05 15 25 35  45 55 65 75
  // out[6]: 06 16 26 36  46 56 66 76
  // out[7]: 07 17 27 37  47 57 67 77
  out[0] = _mm_unpacklo_epi64(b0, b1);
  out[1] = _mm_unpackhi_epi64(b0, b1);
  out[2] = _mm_unpacklo_epi64(b4, b5);
  out[3] = _mm_unpackhi_epi64(b4, b5);
  out[4] = _mm_unpacklo_epi64(b2, b3);
  out[5] = _mm_unpackhi_epi64(b2, b3);
  out[6] = _mm_unpacklo_epi64(b6, b7);
  out[7] = _mm_unpackhi_epi64(b6, b7);
}

LIBDE265_INLINE void filter_mask_calc(__m128i dp, __m128i dq, int beta, __m128i* filter_mask){
  __m128i p4 = dp, p7 = dp, q5 = dq, q6 =dq;
  __m128i tmp_p[2];
  __m128i tmp_q[2];
  __m128i q0 = _mm_set1_epi16(beta);

  static const int8_t mask8_32_even_odd[16] = { 0, 1, 4, 5, 8,  9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 };

  __m128i a_sh, b_sh;
  a_sh = _mm_shuffle_epi8 (p7, *(__m128i*) mask8_32_even_odd); //a0, a2, a4, a6,  a1, a3, a5, a7
  b_sh = _mm_shuffle_epi8 (p4, *(__m128i*) mask8_32_even_odd); //b0, b2, b4, b6,  b1, b3, b5, b7
  tmp_p[0] = _mm_unpacklo_epi16(a_sh, b_sh); //a0, b0, a2, b2, a4, b4, a6, b6
  tmp_p[1] = _mm_unpackhi_epi16(a_sh, b_sh); //a1, b1, a3, b3, a5, b5, a7, b7b

  a_sh = _mm_shuffle_epi8 (q6, *(__m128i*) mask8_32_even_odd); 
  b_sh = _mm_shuffle_epi8 (q5, *(__m128i*) mask8_32_even_odd); 
  tmp_q[0] = _mm_unpacklo_epi16(a_sh, b_sh); 
  tmp_q[1] = _mm_unpackhi_epi16(a_sh, b_sh); 

  p7 = _mm_slli_epi64(tmp_p[0], 32);
  p4 = _mm_srli_epi64(tmp_p[1], 32);
  p7 = _mm_srli_epi64 (p7, 32);
  p4 = _mm_slli_epi64 (p4, 32);
  p7 = _mm_or_si128(p7, p4);

  q6 = _mm_slli_epi64(tmp_q[0], 32);
  q5 = _mm_srli_epi64(tmp_q[1], 32);
  q6 = _mm_srli_epi64 (q6, 32);
  q5 = _mm_slli_epi64 (q5, 32);
  q6 = _mm_or_si128(q6, q5);

  q5 = _mm_add_epi16(p7,q6);

  __m128i q4 = q5, q3 = q5;
  a_sh = _mm_shuffle_epi32 (q3, 216); //a0, a2, a1, a3
  b_sh = _mm_shuffle_epi32 (q4, 216); //b0, b2, b1, b3
  __m128i tmp[2];
  tmp[0] = _mm_unpacklo_epi32(a_sh, b_sh); //a0, b0, a2, b2
  tmp[1] = _mm_unpackhi_epi32(a_sh, b_sh); //a1, b1, a3,  b3
  q4 = _mm_add_epi16(tmp[1], tmp[0]);

  *filter_mask = _mm_cmpgt_epi16(q0, q4);
}

LIBDE265_INLINE void filter_mask1_calc(__m128i dp, int beta, __m128i* filter_mask){
  // dp = vsetq_lane_s16(0, dp, 0);
  // dp = vsetq_lane_s16(1, dp, 1);
  // dp = vsetq_lane_s16(2, dp, 2);
  // dp = vsetq_lane_s16(3, dp, 3);
  // dp = vsetq_lane_s16(4, dp, 4);
  // dp = vsetq_lane_s16(5, dp, 5);
  // dp = vsetq_lane_s16(6, dp, 6);
  // dp = vsetq_lane_s16(7, dp, 7);

  static const int8_t mask8_32_even_odd[16] = { 0, 1, 4, 5, 8,  9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 };
  
  __m128i p4 = dp, p7 = dp;
  __m128i tmp_p[2];
  __m128i q0 = _mm_set1_epi16(beta);

  p7 = _mm_shuffle_epi8 (p7, *(__m128i*) mask8_32_even_odd); //a0, a2, a4, a6,  a1, a3, a5, a7
  p4 = _mm_shuffle_epi8 (p4, *(__m128i*) mask8_32_even_odd); //b0, b2, b4, b6,  b1, b3, b5, b7
  tmp_p[0] = _mm_unpacklo_epi16(p7, p4); //a0, b0, a2, b2, a4, b4, a6, b6
  tmp_p[1] = _mm_unpackhi_epi16(p7, p4); //a1, b1, a3, b3, a5, b5, a7, b7

  p7 = _mm_slli_epi64(tmp_p[0], 32); 
  p4 = _mm_srli_epi64(tmp_p[1], 32);
  p7 = _mm_srli_epi64(p7, 32);
  p4 = _mm_slli_epi64(p4, 32); 
  p7 = _mm_or_si128(p7, p4);

  __m128i q4 = p7, q3 = p7;
  __m128i tmp[2];

  q3 = _mm_shuffle_epi32 (q3, 216); //a0, a2, a1, a3
  q4 = _mm_shuffle_epi32 (q4, 216); //b0, b2, b1, b3
  tmp[0] = _mm_unpacklo_epi32(q3, q4); //a0, b0, a2, b2
  tmp[1] = _mm_unpackhi_epi32(q3, q4); //a1, b1, a3,  b3

  q4 = _mm_add_epi16(tmp[1],tmp[0]);

  *filter_mask = _mm_cmpgt_epi16(q0, q4);

  //std::cout << std::endl;
}


LIBDE265_INLINE __m128i vrshrq(__m128i a, int b) 
{
    __m128i mask, r;
    mask =  _mm_slli_epi16(a, (16 - b)); 
    mask = _mm_srli_epi16(mask, 15); 
    r = _mm_srai_epi16 (a, b);
    return _mm_add_epi16 (r, mask); 
}


LIBDE265_INLINE __m128i vbslq(__m128i a, __m128i b, __m128i c) 
{
    __m128i tmp1, tmp2;
    tmp1 = _mm_and_si128   (a, b);
    tmp2 = _mm_andnot_si128 (a, c);
    return _mm_or_si128 (tmp1, tmp2);
}

LIBDE265_INLINE __m128i vrhaddq(__m128i a, __m128i b) // VRHADD.S16 q0,q0,q0
{
    //no signed average in x86 SIMD, go to unsigned
    __m128i cx8000, au, bu, sum;
    cx8000 = _mm_set1_epi16(-32768); //(int16_t)0x8000
    au = _mm_sub_epi16(a, cx8000); //add 32768
    bu = _mm_sub_epi16(b, cx8000); //add 32768
    sum = _mm_avg_epu16(au, bu);
    return _mm_add_epi16 (sum, cx8000); //sub 32768
}

LIBDE265_INLINE __m128i vhaddq(__m128i a, __m128i b)
{
    //need to avoid internal overflow, will use the (x&y)+((x^y)>>1).
    __m128i tmp1, tmp2;
    tmp1 = _mm_and_si128(a,b);
    tmp2 = _mm_xor_si128(a,b);
    tmp2 = _mm_srai_epi16(tmp2,1);
    return _mm_add_epi16(tmp1,tmp2);
}

LIBDE265_INLINE __m128i vhsubq_u16(__m128i a, __m128i b) // VHSUB.s16 q0,q0,q0
{
    __m128i avg;
    avg = _mm_avg_epu16 (a, b);
    return _mm_sub_epi16(a, avg);
}

LIBDE265_INLINE __m128i vhsubq(__m128i a, __m128i b) // VHSUB.S16 q0,q0,q0
{
    //need to deal with the possibility of internal overflow
    __m128i c8000, au,bu;
    c8000 = _mm_set1_epi16(-32768); //(int16_t)0x8000
    au = _mm_add_epi16( a, c8000);
    bu = _mm_add_epi16( b, c8000);
    return vhsubq_u16(au,bu);
}

LIBDE265_INLINE void luma_strong_filter(__m128i* data, __m128i tc2, __m128i ntc2, __m128i filter_mask){
  __m128i a  = _mm_add_epi16(data[3], data[4]);  //p0+q0
  __m128i b  = _mm_add_epi16(data[1], data[2]);  //p2+p1
  __m128i c  = _mm_add_epi16(data[5], data[6]);  //q1+q2
  __m128i d  = _mm_add_epi16(data[2], data[5]);  //p1+q1
  __m128i e  = _mm_add_epi16(data[0], data[1]);  //p3+q2
  __m128i f  = _mm_add_epi16(data[7], data[6]);  //q3+q2
  __m128i ab = _mm_add_epi16(a, b);
  __m128i ac = _mm_add_epi16(a, c);

  __m128i p[6];

  p[0] = _mm_add_epi16(ab, a);
  p[0] = _mm_add_epi16(p[0], d);
  p[0] = vrshrq(p[0], 3);
  p[0] = _mm_sub_epi16(p[0], data[3]);
  p[0] = _mm_min_epi16(p[0], tc2);
  p[0] = _mm_max_epi16(p[0], ntc2);
  p[0] = _mm_add_epi16(p[0], data[3]);
  data[3] = vbslq(filter_mask, p[0], data[3]);

  p[1] = vrshrq(ab, 2);
  p[1] = _mm_sub_epi16(p[1], data[2]);
  p[1] = _mm_min_epi16(p[1], tc2);
  p[1] = _mm_max_epi16(p[1], ntc2);
  p[1] = _mm_add_epi16(p[1], data[2]);
  data[2] = vbslq(filter_mask, p[1], data[2]);

  p[2] = _mm_slli_epi16(e, 1);
  p[2] = _mm_add_epi16(ab, p[2]);
  p[2] = vrshrq(p[2], 3);
  p[2] = _mm_sub_epi16(p[2], data[1]);
  p[2] = _mm_min_epi16(p[2], tc2);
  p[2] = _mm_max_epi16(p[2], ntc2);
  p[2] = _mm_add_epi16(p[2], data[1]);
  data[1] = vbslq(filter_mask, p[2], data[1]);

  p[3] = _mm_add_epi16(ac, a);
  p[3] = _mm_add_epi16(p[3],d);
  p[3] =  vrshrq(p[3], 3);
  p[3] = _mm_sub_epi16(p[3], data[4]);
  p[3] = _mm_min_epi16(p[3], tc2);
  p[3] = _mm_max_epi16(p[3], ntc2);
  p[3] = _mm_add_epi16(p[3], data[4]);
  data[4] = vbslq(filter_mask, p[3], data[4]);
  
  p[4] = vrshrq(ac, 2);
  p[4] = _mm_sub_epi16(p[4], data[5]);
  p[4] = _mm_min_epi16(p[4], tc2);
  p[4] = _mm_max_epi16(p[4], ntc2);
  p[4] = _mm_add_epi16(p[4], data[5]);
  data[5] = vbslq(filter_mask, p[4], data[5]);

  p[5] = _mm_slli_epi16(f, 1);
  p[5] = _mm_add_epi16(ac, p[5]);
  p[5] = vrshrq(p[5], 3);
  p[5] = _mm_sub_epi16(p[5], data[6]);
  p[5] = _mm_min_epi16(p[5], tc2);
  p[5] = _mm_max_epi16(p[5], ntc2);
  p[5] = _mm_add_epi16(p[5], data[6]);
  data[6] = vbslq(filter_mask, p[5], data[6]);
}


void ff_hevc_loop_filter_luma_8_sse4(uint8_t *_dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int BIT_DEPTH)
{
  int tc2[2], tc25[2];
  bool filter_enable[2], strong_filter_enable[2];
  uint8_t *dst = vertical ? _dst - 4 :  _dst - 4*stride;
//   int tc[2];
//   int beta = _beta << (BIT_DEPTH - 8);
//   tc[0] = _tc[0] << (BIT_DEPTH - 8);
//   tc[1] = _tc[1] << (BIT_DEPTH - 8);

  // std::cout << "vertical: " << vertical << std::endl;
  // std::cout << "origial edge pixels" << std::endl;
  // for (size_t j = 0; j < 8; j++)
  // {
  //   for (size_t i = 0; i < 8; i++)
  //   {
  //     std::cout << (int) dst[j*stride+i] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

    __m128i edge_pixel_16[8];
    uint8_t *src = dst;
    for (int i = 0; i < 8; i++){
        edge_pixel_16[i]  = _mm_loadl_epi64((__m128i*) src);
        edge_pixel_16[i] = _mm_unpacklo_epi8(edge_pixel_16[i], _mm_setzero_si128());
        src += stride;
    }  

  if(vertical)
    transpose8x8(edge_pixel_16, edge_pixel_16);

  __m128i dp = _mm_add_epi16(edge_pixel_16[1], edge_pixel_16[3]);
  __m128i dq = _mm_add_epi16(edge_pixel_16[4], edge_pixel_16[6]);
          dp = _mm_sub_epi16(dp, edge_pixel_16[2]);
          dq = _mm_sub_epi16(dq, edge_pixel_16[5]);
          dp = _mm_sub_epi16(dp, edge_pixel_16[2]);                               //P2-2*P1+P0
          dp = _mm_abs_epi16(dp);
          dq = _mm_sub_epi16(dq, edge_pixel_16[5]);                               //Q2-2*Q1+Q0
          dq = _mm_abs_epi16(dq);

   __m128i dec1 = _mm_add_epi16(dp, dq);
   __m128i filter_mask;
   filter_mask_calc(dp, dq, beta, &filter_mask);

   filter_enable[0] = _mm_extract_epi64(filter_mask, 0);
   filter_enable[1] = _mm_extract_epi64(filter_mask, 1);

   if((!filter_enable[0]) && (!filter_enable[1])) 
   {
        return;
   }

    const int beta_3 = beta >> 3;
    const int beta_2 = beta >> 2;
    tc2[0] = tc[0] << 1; tc2[1] = tc[1] << 1;
    tc25[0]   = ((tc2[0]*2 + tc[0]  + 1) >> 1);
    tc25[1]   = ((tc2[1]*2 + tc[1]  + 1) >> 1);   

    __m128i tc2_array = _mm_set_epi16(tc2[1],tc2[1],tc2[1],tc2[1],tc2[0],tc2[0],tc2[0],tc2[0]);
    __m128i zero = _mm_setzero_si128 ();
    __m128i ntc2_array = _mm_sub_epi16 (zero, tc2_array);
                                                
    dec1 = _mm_slli_epi16(dec1, 1);

    __m128i dp1 = _mm_sub_epi16(edge_pixel_16[0], edge_pixel_16[3]); 
            dp1 = _mm_abs_epi16(dp1);
    __m128i dq1 = _mm_sub_epi16(edge_pixel_16[4], edge_pixel_16[7]); 
            dq1 = _mm_abs_epi16(dq1);
    __m128i dec2 = _mm_add_epi16(dp1, dq1);
    __m128i dec3 = _mm_sub_epi16(edge_pixel_16[3], edge_pixel_16[4]); 
            dec3 = _mm_abs_epi16(dec3);

    strong_filter_enable[0] = _mm_extract_epi16(dec1, 0) < beta_2  && _mm_extract_epi16(dec1, 3) < beta_2 &&
                              _mm_extract_epi16(dec2, 0) < beta_3  && _mm_extract_epi16(dec2, 3) < beta_3 &&
                              _mm_extract_epi16(dec3, 0) < tc25[0] && _mm_extract_epi16(dec3, 3) < tc25[0];

    strong_filter_enable[1] = _mm_extract_epi16(dec1, 4) < beta_2  && _mm_extract_epi16(dec1, 7) < beta_2 &&
                              _mm_extract_epi16(dec2, 4) < beta_3  && _mm_extract_epi16(dec2, 7) < beta_3 &&
                              _mm_extract_epi16(dec3, 4) < tc25[1] && _mm_extract_epi16(dec3, 7) < tc25[1];

    uint16_t strong_filter_mask_l = strong_filter_enable[0] ? 65535 : 0;
    uint16_t strong_filter_mask_h = strong_filter_enable[1] ? 65535 : 0;
    __m128i strong_filter_mask = _mm_set_epi16(
                                            strong_filter_mask_h,strong_filter_mask_h,strong_filter_mask_h,strong_filter_mask_h,
                                            strong_filter_mask_l,strong_filter_mask_l,strong_filter_mask_l,strong_filter_mask_l
                                            );

    __m128i normal_filter_mask = _mm_cmpeq_epi16 (strong_filter_mask, strong_filter_mask); //0xffff
            normal_filter_mask =_mm_andnot_si128 (strong_filter_mask, normal_filter_mask);
            
    strong_filter_mask = _mm_and_si128(strong_filter_mask, filter_mask);
    normal_filter_mask = _mm_and_si128(normal_filter_mask, filter_mask);

    if(strong_filter_enable[0] || strong_filter_enable[1])
        luma_strong_filter(edge_pixel_16, tc2_array, ntc2_array, strong_filter_mask);

  if(!(strong_filter_enable[0] && strong_filter_enable[1]))
  {
        __m128i nd_p_mask, nd_q_mask;
        int belta0 = ((beta + (beta >> 1)) >> 3);
        filter_mask1_calc(dp, belta0, &nd_p_mask);
        filter_mask1_calc(dq, belta0, &nd_q_mask);

        __m128i zero = _mm_setzero_si128();
        __m128i tc_array = _mm_srai_epi16(tc2_array, 1);
        __m128i ntc_array = _mm_sub_epi16(zero, tc_array);
        __m128i tc_1_2_array = _mm_srai_epi16(tc_array, 1);
        __m128i ntc1_2_array  = _mm_sub_epi16(zero, tc_1_2_array);  

        __m128i tc10 = _mm_slli_epi16(tc2_array, 2);
                tc10 = _mm_add_epi16(tc10, tc2_array);

        __m128i delta0 = _mm_sub_epi16(edge_pixel_16[4], edge_pixel_16[3]);
        __m128i q0 = _mm_slli_epi16(delta0, 3);
        delta0 = _mm_add_epi16(q0, delta0);
        q0 = _mm_sub_epi16(edge_pixel_16[5], edge_pixel_16[2]);
        delta0 = _mm_sub_epi16(delta0, q0);
        q0 = _mm_slli_epi16(q0, 1);
        delta0 = _mm_sub_epi16(delta0, q0);
        delta0 = vrshrq(delta0, 4);
        __m128i adelta0 = _mm_abs_epi16(delta0);
        delta0 = _mm_min_epi16(delta0, tc_array);
        delta0 = _mm_max_epi16(delta0, ntc_array);

        __m128i  pixel_filter_mask = _mm_cmpgt_epi16(tc10, adelta0);   

        __m128i deltap1 = vrhaddq(edge_pixel_16[1], edge_pixel_16[3]);
        deltap1 = _mm_sub_epi16(deltap1, edge_pixel_16[2]);
        deltap1 = vhaddq(deltap1, delta0);
        deltap1 = _mm_min_epi16(deltap1, tc_1_2_array);
        deltap1 = _mm_max_epi16(deltap1, ntc1_2_array);

        __m128i p1 = _mm_add_epi16(edge_pixel_16[2], deltap1);
        __m128i up_mask = _mm_and_si128(normal_filter_mask, pixel_filter_mask);
        __m128i up_p_mask = _mm_and_si128(up_mask, nd_p_mask);
        edge_pixel_16[2] = vbslq(up_p_mask, p1, edge_pixel_16[2]);
        p1 = _mm_add_epi16(edge_pixel_16[3], delta0);
        edge_pixel_16[3] = vbslq(up_mask, p1, edge_pixel_16[3]);

        __m128i deltap2 = vrhaddq(edge_pixel_16[4], edge_pixel_16[6]);
        deltap2 = _mm_sub_epi16(deltap2, edge_pixel_16[5]);
        deltap2 = vhsubq(deltap2, delta0);
        deltap2 = _mm_min_epi16(deltap2, tc_1_2_array);
        deltap2 = _mm_max_epi16(deltap2, ntc1_2_array);

        __m128i q1 = _mm_add_epi16(edge_pixel_16[5], deltap2);
        up_mask = _mm_and_si128(normal_filter_mask, pixel_filter_mask);
        __m128i up_q_mask = _mm_and_si128(up_mask, nd_q_mask);
        edge_pixel_16[5] = vbslq(up_q_mask, q1, edge_pixel_16[5]);
        q1 = _mm_sub_epi16(edge_pixel_16[4], delta0);
        edge_pixel_16[4] = vbslq(up_mask, q1, edge_pixel_16[4]);
   }


  if(vertical)
    transpose8x8(edge_pixel_16, edge_pixel_16);

   src = dst;
   for (int i = 0; i < 8; i++)
   {
        __m128i tmp = _mm_packus_epi16(edge_pixel_16[i], edge_pixel_16[i]);    
        _mm_storel_epi64((__m128i *) src, tmp);
        src += stride;
   }
}

LIBDE265_INLINE void Transpose4x8To8x4(const __m128i* in, __m128i* out) {
  // Unpack 16 bit elements. Goes from:
  // in[0]: 00 01 02 03  XX XX XX XX
  // in[1]: 10 11 12 13  XX XX XX XX
  // in[2]: 20 21 22 23  XX XX XX XX
  // in[3]: 30 31 32 33  XX XX XX XX
  // in[4]: 40 41 42 43  XX XX XX XX
  // in[5]: 50 51 52 53  XX XX XX XX
  // in[6]: 60 61 62 63  XX XX XX XX
  // in[7]: 70 71 72 73  XX XX XX XX
  // to:
  // a0:    00 10 01 11  02 12 03 13
  // a1:    20 30 21 31  22 32 23 33
  // a2:    40 50 41 51  42 52 43 53
  // a3:    60 70 61 71  62 72 63 73
  const __m128i a0 = _mm_unpacklo_epi16(in[0], in[1]);
  const __m128i a1 = _mm_unpacklo_epi16(in[2], in[3]);
  const __m128i a2 = _mm_unpacklo_epi16(in[4], in[5]);
  const __m128i a3 = _mm_unpacklo_epi16(in[6], in[7]);

  // Unpack 32 bit elements resulting in:
  // b0: 00 10 20 30  01 11 21 31
  // b1: 40 50 60 70  41 51 61 71
  // b2: 02 12 22 32  03 13 23 33
  // b3: 42 52 62 72  43 53 63 73
  const __m128i b0 = _mm_unpacklo_epi32(a0, a1);
  const __m128i b1 = _mm_unpacklo_epi32(a2, a3);
  const __m128i b2 = _mm_unpackhi_epi32(a0, a1);
  const __m128i b3 = _mm_unpackhi_epi32(a2, a3);

  // Unpack 64 bit elements resulting in:
  // out[0]: 00 10 20 30  40 50 60 70
  // out[1]: 01 11 21 31  41 51 61 71
  // out[2]: 02 12 22 32  42 52 62 72
  // out[3]: 03 13 23 33  43 53 63 73
  out[0] = _mm_unpacklo_epi64(b0, b1);
  out[1] = _mm_unpackhi_epi64(b0, b1);
  out[2] = _mm_unpacklo_epi64(b2, b3);
  out[3] = _mm_unpackhi_epi64(b2, b3);
}

LIBDE265_INLINE void Transpose8x4To4x8(const __m128i* in,  __m128i* out) {
  // Unpack 16 bit elements. Goes from:
  // in[0]: 00 01 02 03  04 05 06 07
  // in[1]: 10 11 12 13  14 15 16 17
  // in[2]: 20 21 22 23  24 25 26 27
  // in[3]: 30 31 32 33  34 35 36 37

  // to:
  // a0:    00 10 01 11  02 12 03 13
  // a1:    20 30 21 31  22 32 23 33
  // a4:    04 14 05 15  06 16 07 17
  // a5:    24 34 25 35  26 36 27 37
  const __m128i a0 = _mm_unpacklo_epi16(in[0], in[1]);
  const __m128i a1 = _mm_unpacklo_epi16(in[2], in[3]);
  const __m128i a4 = _mm_unpackhi_epi16(in[0], in[1]);
  const __m128i a5 = _mm_unpackhi_epi16(in[2], in[3]);

  // Unpack 32 bit elements resulting in:
  // b0: 00 10 20 30  01 11 21 31
  // b2: 04 14 24 34  05 15 25 35
  // b4: 02 12 22 32  03 13 23 33
  // b6: 06 16 26 36  07 17 27 37
  const __m128i b0 = _mm_unpacklo_epi32(a0, a1);
  const __m128i b2 = _mm_unpacklo_epi32(a4, a5);
  const __m128i b4 = _mm_unpackhi_epi32(a0, a1);
  const __m128i b6 = _mm_unpackhi_epi32(a4, a5);

  // Unpack 64 bit elements resulting in:
  // out[0]: 00 10 20 30  XX XX XX XX
  // out[1]: 01 11 21 31  XX XX XX XX
  // out[2]: 02 12 22 32  XX XX XX XX
  // out[3]: 03 13 23 33  XX XX XX XX
  // out[4]: 04 14 24 34  XX XX XX XX
  // out[5]: 05 15 25 35  XX XX XX XX
  // out[6]: 06 16 26 36  XX XX XX XX
  // out[7]: 07 17 27 37  XX XX XX XX
  const __m128i zeros = _mm_setzero_si128();
  out[0] = _mm_unpacklo_epi64(b0, zeros);
  out[1] = _mm_unpackhi_epi64(b0, zeros);
  out[2] = _mm_unpacklo_epi64(b4, zeros);
  out[3] = _mm_unpackhi_epi64(b4, zeros);
  out[4] = _mm_unpacklo_epi64(b2, zeros);
  out[5] = _mm_unpackhi_epi64(b2, zeros);
  out[6] = _mm_unpacklo_epi64(b6, zeros);
  out[7] = _mm_unpackhi_epi64(b6, zeros);
}

void ff_hevc_loop_filter_chroma_8_sse4(uint8_t *_dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitdepth)
{

  if ((tc[0]+tc[1])==0)
  {
    return;
  }
  
  uint8_t *dst = vertical ? _dst - 2 :  _dst - 2*stride;
  __m128i edge_pixel16[4];
  uint8_t *src = dst;

  __m128i tcc = _mm_set_epi16(tc[1],tc[1],tc[1],tc[1],tc[0],tc[0],tc[0],tc[0]);
  __m128i ntcc = _mm_set_epi16(-tc[1],-tc[1],-tc[1],-tc[1],-tc[0],-tc[0],-tc[0],-tc[0]);

  if(vertical)
  {
    __m128i input[8];
    for (int i = 0; i < 8; i++)
    {
        __m128i tmp = _mm_loadl_epi64((__m128i *) src); 
        input[i] = _mm_unpacklo_epi8(tmp, _mm_setzero_si128());
        src += stride;
    }
    Transpose4x8To8x4(input, edge_pixel16);
    
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      __m128i tmp = _mm_loadl_epi64((__m128i *) src); 
      edge_pixel16[i] = _mm_unpacklo_epi8(tmp, _mm_setzero_si128());
      src += stride;
    }
  }

  __m128i delta = _mm_sub_epi16(edge_pixel16[2], edge_pixel16[1]);
  delta = _mm_slli_epi16(delta, 2);
  delta = _mm_add_epi16(delta, edge_pixel16[0]);
  delta = _mm_sub_epi16(delta, edge_pixel16[3]);
  delta = vrshrq(delta, 3);
  delta = _mm_min_epi16(delta, tcc);
  delta = _mm_max_epi16(delta, ntcc);
  edge_pixel16[1] = _mm_add_epi16(edge_pixel16[1], delta);
  edge_pixel16[2] = _mm_sub_epi16(edge_pixel16[2], delta);


  if (vertical)
  {
    __m128i output[8];
    Transpose8x4To4x8(edge_pixel16, output);
    for (int i = 0; i < 8; i++)
    {
      __m128i tmp = _mm_packus_epi16(output[i], output[i]);
      int32_t val = _mm_cvtsi128_si32(tmp); 
      memcpy(dst, &val, sizeof(val));
      dst += stride;
    }
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      __m128i tmp = _mm_packus_epi16(edge_pixel16[i], edge_pixel16[i]);
      _mm_storel_epi64((__m128i *) dst, tmp);
      dst += stride;
    }
  }
  
}

#endif