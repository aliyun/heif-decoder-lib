#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>

#include <cstring>
#include <arm_neon.h>
#include "./libde265/util.h"
#include "neon_dbk.h"

LIBDE265_INLINE void filter_mask_calc(int16x8_t dp, int16x8_t dq, int beta, uint16x8_t* filter_mask){
  int16x8_t p4 = dp, p7 = dp, q5 = dq, q6 =dq;
  int16x8x2_t tmp_p;
  int16x8x2_t tmp_q;
  int16x8_t q0 = vdupq_n_s16(beta);

  tmp_p = vtrnq_s16(p7, p4);
  tmp_q = vtrnq_s16(q6, q5);

  p7 = vreinterpretq_s16_u64(vshlq_n_u64(vreinterpretq_u64_s16(tmp_p.val[0]),32));
  p4 = vreinterpretq_s16_u64(vshrq_n_u64(vreinterpretq_u64_s16(tmp_p.val[1]),32));
  p7 = vreinterpretq_s16_u64(vshrq_n_u64(vreinterpretq_u64_s16(p7),32));
  p4 = vreinterpretq_s16_u64(vshlq_n_u64(vreinterpretq_u64_s16(p4),32));
  p7 = vorrq_s16(p7, p4);

  q6 = vreinterpretq_s16_u64(vshlq_n_u64(vreinterpretq_u64_s16(tmp_q.val[0]),32));
  q5 = vreinterpretq_s16_u64(vshrq_n_u64(vreinterpretq_u64_s16(tmp_q.val[1]),32));
  q6 = vreinterpretq_s16_u64(vshrq_n_u64(vreinterpretq_u64_s16(q6),32));
  q5 = vreinterpretq_s16_u64(vshlq_n_u64(vreinterpretq_u64_s16(q5),32));
  q6 = vorrq_s16(q6, q5);
  q5 = vaddq_s16(p7,q6); 

  int16x8_t q4 = q5, q3 = q5;
  int32x4x2_t tmp = vtrnq_s32(vreinterpretq_s32_s16(q3),vreinterpretq_s32_s16(q4));
  q3 = vreinterpretq_s16_s32(tmp.val[0]);
  q4 = vreinterpretq_s16_s32(tmp.val[1]);
  q4 = vaddq_s16(q4,q3);
 
  *filter_mask = vcgtq_s16(q0, q4);

}

LIBDE265_INLINE void filter_mask1_calc(int16x8_t dp, int beta, uint16x8_t* filter_mask){
  // dp = vsetq_lane_s16(0, dp, 0);
  // dp = vsetq_lane_s16(1, dp, 1);
  // dp = vsetq_lane_s16(2, dp, 2);
  // dp = vsetq_lane_s16(3, dp, 3);
  // dp = vsetq_lane_s16(4, dp, 4);
  // dp = vsetq_lane_s16(5, dp, 5);
  // dp = vsetq_lane_s16(6, dp, 6);
  // dp = vsetq_lane_s16(7, dp, 7);
  
  int16x8_t p4 = dp, p7 = dp;
  int16x8x2_t tmp_p;
  int16x8_t q0 = vdupq_n_s16(beta);

  tmp_p = vtrnq_s16(p7, p4);

  p7 = vreinterpretq_s16_u64(vshlq_n_u64(vreinterpretq_u64_s16(tmp_p.val[0]),32));
  p4 = vreinterpretq_s16_u64(vshrq_n_u64(vreinterpretq_u64_s16(tmp_p.val[1]),32));
  p7 = vreinterpretq_s16_u64(vshrq_n_u64(vreinterpretq_u64_s16(p7),32));
  p4 = vreinterpretq_s16_u64(vshlq_n_u64(vreinterpretq_u64_s16(p4),32));
  p7 = vorrq_s16(p7, p4);

  int16x8_t q4 = p7, q3 = p7;
  int32x4x2_t tmp = vtrnq_s32(vreinterpretq_s32_s16(q3),vreinterpretq_s32_s16(q4));
  q3 = vreinterpretq_s16_s32(tmp.val[0]);
  q4 = vreinterpretq_s16_s32(tmp.val[1]);
  q4 = vaddq_s16(q4,q3);

  *filter_mask = vcgtq_s16(q0, q4);

  //std::cout << std::endl;
}

LIBDE265_INLINE void luma_strong_filter(int16x8_t* data, int16x8_t tc2, int16x8_t ntc2, uint16x8_t filter_mask){
  int16x8_t a = vaddq_s16(data[3], data[4]);  //p0+q0
  int16x8_t b = vaddq_s16(data[1], data[2]);  //p2+p1
  int16x8_t c = vaddq_s16(data[5], data[6]);  //q1+q2
  int16x8_t d = vaddq_s16(data[2], data[5]);  //p1+q1
  int16x8_t e = vaddq_s16(data[0], data[1]);  //p3+q2
  int16x8_t f = vaddq_s16(data[7], data[6]);  //q3+q2
  int16x8_t ab = vaddq_s16(a, b);
  int16x8_t ac = vaddq_s16(a, c);

  int16x8_t p[6];

  p[0] = vaddq_s16(ab, a);
  p[0] = vaddq_s16(p[0], d);
  p[0] = vrshrq_n_s16(p[0], 3);
  p[0] = vsubq_s16(p[0], data[3]);
  p[0] = vminq_s16(p[0], tc2);
  p[0] = vmaxq_s16(p[0], ntc2);
  p[0] = vaddq_s16(p[0], data[3]);
  data[3] = vbslq_s16(filter_mask, p[0], data[3]);

  p[1] = vrshrq_n_s16(ab, 2);
  p[1] = vsubq_s16(p[1], data[2]);
  p[1] = vminq_s16(p[1], tc2);
  p[1] = vmaxq_s16(p[1], ntc2);
  p[1] = vaddq_s16(p[1], data[2]);
  data[2] = vbslq_s16(filter_mask, p[1], data[2]);

  p[2] = vshlq_n_s16(e, 1);
  p[2] = vaddq_s16(ab, p[2]);
  p[2] = vrshrq_n_s16(p[2], 3);
  p[2] = vsubq_s16(p[2], data[1]);
  p[2] = vminq_s16(p[2], tc2);
  p[2] = vmaxq_s16(p[2], ntc2);
  p[2] = vaddq_s16(p[2], data[1]);
  data[1] = vbslq_s16(filter_mask, p[2], data[1]);

  p[3] = vaddq_s16(ac, a);
  p[3] = vaddq_s16(p[3],d);
  p[3] =  vrshrq_n_s16(p[3], 3);
  p[3] = vsubq_s16(p[3], data[4]);
  p[3] = vminq_s16(p[3], tc2);
  p[3] = vmaxq_s16(p[3], ntc2);
  p[3] = vaddq_s16(p[3], data[4]);
  data[4] = vbslq_s16(filter_mask, p[3], data[4]);


  p[4] = vrshrq_n_s16(ac, 2);
  p[4] = vsubq_s16(p[4], data[5]);
  p[4] = vminq_s16(p[4], tc2);
  p[4] = vmaxq_s16(p[4], ntc2);
  p[4] = vaddq_s16(p[4], data[5]);
  data[5] = vbslq_s16(filter_mask, p[4], data[5]);

  p[5] = vshlq_n_s16(f, 1);
  p[5] = vaddq_s16(ac, p[5]);
  p[5] = vrshrq_n_s16(p[5], 3);
  p[5] = vsubq_s16(p[5], data[6]);
  p[5] = vminq_s16(p[5], tc2);
  p[5] = vmaxq_s16(p[5], ntc2);
  p[5] = vaddq_s16(p[5], data[6]);
  data[6] = vbslq_s16(filter_mask, p[5], data[6]);
}


LIBDE265_INLINE void  transpose8x8(uint8x8_t a[8]) {
  const uint8x16x2_t b0 =
      vtrnq_u8(vcombine_u8(a[0], a[4]), vcombine_u8(a[1], a[5]));
  const uint8x16x2_t b1 =
      vtrnq_u8(vcombine_u8(a[2], a[6]), vcombine_u8(a[3], a[7]));

  const uint16x8x2_t c0 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[0]),
                                    vreinterpretq_u16_u8(b1.val[0]));
  const uint16x8x2_t c1 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[1]),
                                    vreinterpretq_u16_u8(b1.val[1]));

  const uint32x4x2_t d0 = vuzpq_u32(vreinterpretq_u32_u16(c0.val[0]),
                                    vreinterpretq_u32_u16(c1.val[0]));
  const uint32x4x2_t d1 = vuzpq_u32(vreinterpretq_u32_u16(c0.val[1]),
                                    vreinterpretq_u32_u16(c1.val[1]));

  a[0] = vreinterpret_u8_u32(vget_low_u32(d0.val[0]));
  a[1] = vreinterpret_u8_u32(vget_high_u32(d0.val[0]));
  a[2] = vreinterpret_u8_u32(vget_low_u32(d1.val[0]));
  a[3] = vreinterpret_u8_u32(vget_high_u32(d1.val[0]));
  a[4] = vreinterpret_u8_u32(vget_low_u32(d0.val[1]));
  a[5] = vreinterpret_u8_u32(vget_high_u32(d0.val[1]));
  a[6] = vreinterpret_u8_u32(vget_low_u32(d1.val[1]));
  a[7] = vreinterpret_u8_u32(vget_high_u32(d1.val[1]));
}


void ff_hevc_loop_filter_luma_8_neon(uint8_t *_dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int BIT_DEPTH)
{
  int tc2[2], tc25[2];
  bool filter_disable[2], strong_filter_enable[2];
  uint8_t *dst = vertical ? _dst - 4 :  _dst - 4*stride;
  //int tc[2];
  // int beta = _beta << (BIT_DEPTH - 8);
  // tc[0] = _tc[0] << (BIT_DEPTH - 8);
  // tc[1] = _tc[1] << (BIT_DEPTH - 8);

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

  uint8x8_t edge_pixel[8];
  int16x8_t edge_pixel_16[8];
  uint8_t *src = dst;
  for (int i = 0; i < 8; i++){
    edge_pixel[i]  = vld1_u8(src);
    src += stride;
  }  
  
  //unit_test();

  if(vertical)
    transpose8x8(edge_pixel);

  // std::cout << "transpose edge pixels" << std::endl;
  // for (size_t j = 0; j < 8; j++)
  // {
  //   std::cout << (int) vget_lane_u8(edge_pixel[j], 0) << " ";
  //   std::cout << (int) vget_lane_u8(edge_pixel[j], 1) << " ";
  //   std::cout << (int) vget_lane_u8(edge_pixel[j], 2) << " ";
  //   std::cout << (int) vget_lane_u8(edge_pixel[j], 3) << " ";
  //   std::cout << (int) vget_lane_u8(edge_pixel[j], 4) << " ";
  //   std::cout << (int) vget_lane_u8(edge_pixel[j], 5) << " ";
  //   std::cout << (int) vget_lane_u8(edge_pixel[j], 6) << " ";
  //   std::cout << (int) vget_lane_u8(edge_pixel[j], 7) << " ";
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;


  src = dst;
  for (int i = 0; i < 8; i++)
    edge_pixel_16[i]  = vreinterpretq_s16_u16(vmovl_u8(edge_pixel[i]));


  int16x8_t dp = vaddq_s16(edge_pixel_16[1], edge_pixel_16[3]);
  int16x8_t dq = vaddq_s16(edge_pixel_16[4], edge_pixel_16[6]);
  dp = vsubq_s16(dp, edge_pixel_16[2]);
  dq = vsubq_s16(dq, edge_pixel_16[5]);
  dp = vabdq_s16(dp, edge_pixel_16[2]);                               //P2-2*P1+P0
  dq = vabdq_s16(dq, edge_pixel_16[5]);                               //Q2-2*Q1+Q0
  int16x8_t dec1 = vaddq_s16(dp, dq);
  uint16x8_t filter_mask;
  filter_mask_calc(dp, dq, beta, &filter_mask);

  filter_disable[0]  = !vgetq_lane_u64(vreinterpretq_u64_u16(filter_mask), 0);
  filter_disable[1]  = !vgetq_lane_u64(vreinterpretq_u64_u16(filter_mask), 1);

  if(filter_disable[0] && filter_disable[1]) 
  {
  // std::cout << "filtered edge pixels" << std::endl;
  // for (size_t j = 0; j < 8; j++)
  // {
  //   for (size_t i = 0; i < 8; i++)
  //   {
  //     std::cout << (int) dst[j*stride+i] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;
    return;
  }

  const int beta_3 = beta >> 3;
  const int beta_2 = beta >> 2;
  tc2[0] = tc[0] << 1; tc2[1] = tc[1] << 1;
  tc25[0]   = ((tc2[0]*2 + tc[0]  + 1) >> 1);
  tc25[1]   = ((tc2[1]*2 + tc[1]  + 1) >> 1);   
  int16x4_t tc2_0 = vdup_n_s16(tc2[0]);                        
  int16x4_t tc2_1 = vdup_n_s16(tc2[1]);
  int16x8_t tc2_array   = vcombine_s16(tc2_0, tc2_1);                                           
  int16x8_t ntc2_array  = vnegq_s16(tc2_array);                                                   
  dec1 = vshlq_n_s16(dec1, 1);
  int16x8_t dp1 = vabdq_s16(edge_pixel_16[0], edge_pixel_16[3]);
  int16x8_t dq1 = vabdq_s16(edge_pixel_16[4], edge_pixel_16[7]);
  int16x8_t dec2 = vaddq_s16(dp1, dq1);
  int16x8_t dec3 = vabdq_s16(edge_pixel_16[3], edge_pixel_16[4]);
  strong_filter_enable[0] = vgetq_lane_s16(dec1, 0) < beta_2  && vgetq_lane_s16(dec1, 3) < beta_2 &&
                            vgetq_lane_s16(dec2, 0) < beta_3  && vgetq_lane_s16(dec2, 3) < beta_3 &&
                            vgetq_lane_s16(dec3, 0) < tc25[0] && vgetq_lane_s16(dec3, 3) < tc25[0];

  strong_filter_enable[1] = vgetq_lane_s16(dec1, 4) < beta_2  && vgetq_lane_s16(dec1, 7) < beta_2 &&
                            vgetq_lane_s16(dec2, 4) < beta_3  && vgetq_lane_s16(dec2, 7) < beta_3 &&
                            vgetq_lane_s16(dec3, 4) < tc25[1] && vgetq_lane_s16(dec3, 7) < tc25[1];
  uint16x4_t strong_filter_mask_l =  strong_filter_enable[0] ? vdup_n_u16(65535) : vdup_n_u16(0);
  uint16x4_t strong_filter_mask_h =  strong_filter_enable[1] ? vdup_n_u16(65535) : vdup_n_u16(0);
  uint16x8_t strong_filter_mask = vcombine_u16(strong_filter_mask_l, strong_filter_mask_h);
  uint16x8_t normal_filter_mask = vmvnq_u16(strong_filter_mask);

  strong_filter_mask = vandq_u16(strong_filter_mask, filter_mask);
  normal_filter_mask = vandq_u16(normal_filter_mask, filter_mask);

  if(strong_filter_enable[0] || strong_filter_enable[1])
    luma_strong_filter(edge_pixel_16, tc2_array, ntc2_array, strong_filter_mask);

  if(!(strong_filter_enable[0] && strong_filter_enable[1])){
 
    uint16x8_t nd_p_mask, nd_q_mask;
    int belta0 = ((beta + (beta >> 1)) >> 3);
    filter_mask1_calc(dp, belta0, &nd_p_mask);
    filter_mask1_calc(dq, belta0, &nd_q_mask);

    int16x8_t tc_array = vshrq_n_s16(tc2_array, 1);
    int16x8_t ntc_array = vnegq_s16(tc_array);
    int16x8_t tc_1_2_array   = vshrq_n_s16(tc_array, 1);                         
    int16x8_t ntc1_2_array  = vnegq_s16(tc_1_2_array);  

    int16x8_t tc10 = vshlq_n_s16(tc2_array, 2);
    tc10 = vaddq_s16(tc10, tc2_array);

    int16x8_t delta0 = vsubq_s16(edge_pixel_16[4], edge_pixel_16[3]);
    int16x8_t q0 = vshlq_n_s16(delta0, 3);
    delta0 = vaddq_s16(q0, delta0);
    q0 = vsubq_s16(edge_pixel_16[5], edge_pixel_16[2]);
    delta0 = vsubq_s16(delta0, q0);
    q0 = vshlq_n_s16(q0, 1);
    delta0 = vsubq_s16(delta0, q0);
    delta0 = vrshrq_n_s16(delta0, 4);
    int16x8_t adelta0 = vabsq_s16(delta0);
    delta0 = vminq_s16(delta0, tc_array);
    delta0 = vmaxq_s16(delta0, ntc_array);

    uint16x8_t  pixel_filter_mask = vcgtq_s16(tc10, adelta0);   

    int16x8_t deltap1 = vrhaddq_s16(edge_pixel_16[1], edge_pixel_16[3]);
    deltap1 = vsubq_s16(deltap1, edge_pixel_16[2]);
    deltap1 = vhaddq_s16(deltap1, delta0);
    deltap1 = vminq_s16(deltap1, tc_1_2_array);
    deltap1 = vmaxq_s16(deltap1, ntc1_2_array);

    int16x8_t p1 = vaddq_s16(edge_pixel_16[2], deltap1);
    uint16x8_t up_mask = vandq_u16(normal_filter_mask, pixel_filter_mask);
    uint16x8_t up_p_mask = vandq_u16(up_mask, nd_p_mask);
    edge_pixel_16[2] = vbslq_s16(up_p_mask, p1, edge_pixel_16[2]);
    p1 = vaddq_s16(edge_pixel_16[3], delta0);
    edge_pixel_16[3] = vbslq_s16(up_mask, p1, edge_pixel_16[3]);


    int16x8_t deltap2 = vrhaddq_s16(edge_pixel_16[4], edge_pixel_16[6]);
    deltap2 = vsubq_s16(deltap2, edge_pixel_16[5]);
    deltap2 = vhsubq_s16(deltap2, delta0);
    deltap2 = vminq_s16(deltap2, tc_1_2_array);
    deltap2 = vmaxq_s16(deltap2, ntc1_2_array);

    int16x8_t q1 = vaddq_s16(edge_pixel_16[5], deltap2);
    up_mask = vandq_u16(normal_filter_mask, pixel_filter_mask);
    uint16x8_t up_q_mask = vandq_u16(up_mask, nd_q_mask);
    edge_pixel_16[5] = vbslq_s16(up_q_mask, q1, edge_pixel_16[5]);
    q1 = vsubq_s16(edge_pixel_16[4], delta0);
    edge_pixel_16[4] = vbslq_s16(up_mask, q1, edge_pixel_16[4]);

  }

  for (int i = 0; i < 8; i++)
    edge_pixel[i] = vqmovun_s16(edge_pixel_16[i]);
  

  if(vertical)
    transpose8x8(edge_pixel);

  src = dst;
  for (int i = 0; i < 8; i++){
    vst1_u8(src, edge_pixel[i]);
    src += stride;
  }
  

  // std::cout << "filtered edge pixels" << std::endl;
  // for (size_t j = 0; j < 8; j++)
  // {
  //   for (size_t i = 0; i < 8; i++)
  //   {
  //     std::cout << (int) dst[j*stride+i] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;


  //std::cout << std::endl;
}

LIBDE265_INLINE void Transpose4x4(int16x4_t a[4]) {
  // b:
  // 00 10 02 12
  // 01 11 03 13
  const int16x4x2_t b = vtrn_s16(a[0], a[1]);
  // c:
  // 20 30 22 32
  // 21 31 23 33
  const int16x4x2_t c = vtrn_s16(a[2], a[3]);
  // d:
  // 00 10 20 30
  // 02 12 22 32
  const int32x2x2_t d =
      vtrn_s32(vreinterpret_s32_s16(b.val[0]), vreinterpret_s32_s16(c.val[0]));
  // e:
  // 01 11 21 31
  // 03 13 23 33
  const int32x2x2_t e =
      vtrn_s32(vreinterpret_s32_s16(b.val[1]), vreinterpret_s32_s16(c.val[1]));
  a[0] = vreinterpret_s16_s32(d.val[0]);
  a[1] = vreinterpret_s16_s32(e.val[0]);
  a[2] = vreinterpret_s16_s32(d.val[1]);
  a[3] = vreinterpret_s16_s32(e.val[1]);
}



#ifdef OPT_4
LIBDE265_INLINE void Transpose4x8To8x4(const int16x8_t in[8],
                                             int16x8_t out[4]) {
  // Swap 16 bit elements. Goes from:
  // a0: 00 01 02 03
  // a1: 10 11 12 13
  // a2: 20 21 22 23
  // a3: 30 31 32 33
  // a4: 40 41 42 43
  // a5: 50 51 52 53
  // a6: 60 61 62 63
  // a7: 70 71 72 73
  // to:
  // b0.val[0]: 00 10 02 12
  // b0.val[1]: 01 11 03 13
  // b1.val[0]: 20 30 22 32
  // b1.val[1]: 21 31 23 33
  // b2.val[0]: 40 50 42 52
  // b2.val[1]: 41 51 43 53
  // b3.val[0]: 60 70 62 72
  // b3.val[1]: 61 71 63 73

  int16x4x2_t a0 = vtrn_s16(vget_low_s16(in[0]), vget_low_s16(in[1]));
  int16x4x2_t a1 = vtrn_s16(vget_low_s16(in[2]), vget_low_s16(in[3]));
  int16x4x2_t a2 = vtrn_s16(vget_low_s16(in[4]), vget_low_s16(in[5]));
  int16x4x2_t a3 = vtrn_s16(vget_low_s16(in[6]), vget_low_s16(in[7]));

  // Swap 32 bit elements resulting in:
  // c0.val[0]: 00 10 20 30
  // c0.val[1]: 02 12 22 32
  // c1.val[0]: 01 11 21 31
  // c1.val[1]: 03 13 23 33
  // c2.val[0]: 40 50 60 70
  // c2.val[1]: 42 52 62 72
  // c3.val[0]: 41 51 61 71
  // c3.val[1]: 43 53 63 73

  int32x2x2_t b0 = vtrn_s32(vreinterpret_s32_s16(a0.val[0]), vreinterpret_s32_s16(a1.val[0]));
  int32x2x2_t b1 = vtrn_s32(vreinterpret_s32_s16(a0.val[1]), vreinterpret_s32_s16(a1.val[1]));
  int32x2x2_t b2 = vtrn_s32(vreinterpret_s32_s16(a2.val[0]), vreinterpret_s32_s16(a3.val[0]));
  int32x2x2_t b3 = vtrn_s32(vreinterpret_s32_s16(a2.val[1]), vreinterpret_s32_s16(a3.val[1]));
  // Swap 64 bit elements resulting in:
  // o0: 00 10 20 30 40 50 60 70
  // o1: 01 11 21 31 41 51 61 71
  // o2: 02 12 22 32 42 52 62 72
  // o3: 03 13 23 33 43 53 63 73

  out[0] = vcombine_s16(vreinterpret_s16_s32(b0.val[0]), vreinterpret_s16_s32(b2.val[0]));
  out[1] = vcombine_s16(vreinterpret_s16_s32(b1.val[0]), vreinterpret_s16_s32(b3.val[0]));
  out[2] = vcombine_s16(vreinterpret_s16_s32(b0.val[1]), vreinterpret_s16_s32(b2.val[1]));
  out[3] = vcombine_s16(vreinterpret_s16_s32(b1.val[1]), vreinterpret_s16_s32(b3.val[1]));
}

LIBDE265_INLINE void Transpose8x4To4x8(const int16x8_t in[4],
                                             int16x8_t out[8]) {
  // Swap 16 bit elements. Goes from:
  // a0: 00 01 02 03 04 05 06 07
  // a1: 10 11 12 13 14 15 16 17
  // a2: 20 21 22 23 24 25 26 27
  // a3: 30 31 32 33 34 35 36 37
  // to:
  // b0.val[0]: 00 10 02 12 04 14 06 16
  // b0.val[1]: 01 11 03 13 05 15 07 17
  // b1.val[0]: 20 30 22 32 24 34 26 36
  // b1.val[1]: 21 31 23 33 25 35 27 37
  const int16x8x2_t a0 = vtrnq_s16(in[0], in[1]);
  const int16x8x2_t a1 = vtrnq_s16(in[2], in[3]);

  // Swap 32 bit elements resulting in:
  // c0.val[0]: 00 10 20 30 04 14 24 34
  // c0.val[1]: 02 12 22 32 06 16 26 36
  // c1.val[0]: 01 11 21 31 05 15 25 35
  // c1.val[1]: 03 13 23 33 07 17 27 37
  const int32x4x2_t b0 = vtrnq_s32(vreinterpretq_s32_s16(a0.val[0]), vreinterpretq_s32_s16(a1.val[0]));
  const int32x4x2_t b1 = vtrnq_s32(vreinterpretq_s32_s16(a0.val[1]), vreinterpretq_s32_s16(a1.val[1]));

  // The upper 8 bytes are don't cares.
  // out[0]: 00 10 20 30 04 14 24 34
  // out[1]: 01 11 21 31 05 15 25 35
  // out[2]: 02 12 22 32 06 16 26 36
  // out[3]: 03 13 23 33 07 17 27 37
  // out[4]: 04 14 24 34 04 14 24 34
  // out[5]: 05 15 25 35 05 15 25 35
  // out[6]: 06 16 26 36 06 16 26 36
  // out[7]: 07 17 27 37 07 17 27 37
  out[0] = vreinterpretq_s16_s32(b0.val[0]);
  out[1] = vreinterpretq_s16_s32(b1.val[0]);
  out[2] = vreinterpretq_s16_s32(b0.val[1]);
  out[3] = vreinterpretq_s16_s32(b1.val[1]);
  out[4] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b0.val[0]), vget_high_s32(b0.val[0])));
  out[5] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b1.val[0]), vget_high_s32(b1.val[0])));
  out[6] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b0.val[1]), vget_high_s32(b0.val[1])));
  out[7] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b1.val[1]), vget_high_s32(b1.val[1])));
}

void ff_hevc_loop_filter_chroma_8_neon(uint8_t *_dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitdepth){

  if ((tc[0]+tc[1])==0)
  {
    return;
  }
  
  uint8_t *dst = vertical ? _dst - 2 :  _dst - 2*stride;
  int16x8_t edge_pixel16[4];
  uint8_t *src = dst;

  int16x4_t tc0 = vdup_n_s16(tc[0]);
  int16x4_t ntc0 = vneg_s16(tc0);
  int16x4_t tc1 = vdup_n_s16(tc[1]);
  int16x4_t ntc1 = vneg_s16(tc1);
  int16x8_t tcc = vcombine_s16(tc0, tc1);
  int16x8_t ntcc = vcombine_s16(ntc0, ntc1);

  if(vertical)
  {
    int16x8_t input[8];
    for (int i = 0; i < 8; i++)
    {
      uint8x8_t tmp  = vld1_u8(src);
      input[i] = vreinterpretq_s16_u16(vmovl_u8(tmp));
      src += stride;
    }
    Transpose4x8To8x4(input, edge_pixel16);
    
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      uint8x8_t tmp  = vld1_u8(src);
      edge_pixel16[i] = vreinterpretq_s16_u16(vmovl_u8(tmp));
      src += stride;
    }
  }

  int16x8_t delta = vsubq_s16(edge_pixel16[2], edge_pixel16[1]);
  delta = vshlq_n_s16(delta, 2);
  delta = vaddq_s16(delta, edge_pixel16[0]);
  delta = vsubq_s16(delta, edge_pixel16[3]);
  delta = vrshrq_n_s16(delta, 3);
  delta = vminq_s16(delta, tcc);
  delta = vmaxq_s16(delta, ntcc);
  edge_pixel16[1] = vaddq_s16(edge_pixel16[1], delta);
  edge_pixel16[2] = vsubq_s16(edge_pixel16[2], delta);


  if (vertical)
  {
    int16x8_t output[8];
    Transpose8x4To4x8(edge_pixel16, output);
    for (int i = 0; i < 8; i++)
    {
      int8x8_t tmp = vreinterpret_s8_u8(vqmovun_s16(output[i]));
      int32_t val = vget_lane_s32(vreinterpret_s32_s8(tmp), 0);
      memcpy(dst, &val, sizeof(val));
      dst += stride;
    }
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      uint8x8_t tmp = vqmovun_s16(edge_pixel16[i]);
      vst1_u8(dst, tmp);
      dst += stride;
    }
  }
  
}
#else
void ff_hevc_loop_filter_chroma_8_neon(uint8_t *_dst, bool vertical, ptrdiff_t stride, int _tc, bool filterP, bool filterQ, int bitdepth){
  uint8_t *dst = vertical ? _dst - 2 :  _dst - 2*stride;
  
  int16x4_t edge_pixel_16[4];
  uint8_t *src = dst;
  int16x4_t tc = vdup_n_s16(_tc);
  int16x4_t ntc = vneg_s16(tc);
  for (int i = 0; i < 4; i++){
    uint8x8_t tmp  = vld1_u8(src);
    edge_pixel_16[i] = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(tmp)));
    src += stride;
  }  

  if(vertical)
    Transpose4x4(edge_pixel_16);

  int16x4_t delta = vsub_s16(edge_pixel_16[2], edge_pixel_16[1]);
  delta = vshl_n_s16(delta, 2);
  delta = vadd_s16(delta, edge_pixel_16[0]);
  delta = vsub_s16(delta, edge_pixel_16[3]);
  delta = vrshr_n_s16(delta, 3);
  delta = vmin_s16(delta, tc);
  delta = vmax_s16(delta, ntc);
  edge_pixel_16[1] = vadd_s16(edge_pixel_16[1], delta);
  edge_pixel_16[2] = vsub_s16(edge_pixel_16[2], delta);

  if(vertical)
    Transpose4x4(edge_pixel_16);

  for (int i = 0; i < 4; i++)
  {
    int16x8_t tmp16 = vcombine_s16(edge_pixel_16[i], edge_pixel_16[i]);
    int8x8_t tmp8 = vreinterpret_s8_u8(vqmovun_s16(tmp16));
    int32_t val = vget_lane_s32(vreinterpret_s32_s8(tmp8), 0);
    memcpy(dst, &val, sizeof(val));
    dst += stride;
  }
}
#endif


void ff_hevc_sao_band_filter_8_neon(uint8_t *_dst, int stride_dst, /*const*/ uint8_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, int saoLeftClass, const int8_t *saoOffsetVal, int bandShift, const int maxPixelValue){

  uint8_t* src = _src;
  uint8_t* dst = _dst;

  int16x8_t r0, r1, r2, r3;
  int16x8_t x0, x1, x2, x3;
  uint16x8_t mask0, mask1, mask2, mask3;
  int16x8_t sao0, sao1, sao2, sao3;
  uint8x8_t src8;
  int16x8_t src16;
  int16x8_t src2;

  int16x8_t zeros = vdupq_n_s16(0);

  r0 = vdupq_n_s16((saoLeftClass    ) & 31);
  r1 = vdupq_n_s16((saoLeftClass + 1) & 31);
  r2 = vdupq_n_s16((saoLeftClass + 2) & 31);
  r3 = vdupq_n_s16((saoLeftClass + 3) & 31);
  sao0 = vdupq_n_s16(saoOffsetVal[0]);
  sao1 = vdupq_n_s16(saoOffsetVal[1]);
  sao2 = vdupq_n_s16(saoOffsetVal[2]);
  sao3 = vdupq_n_s16(saoOffsetVal[3]);

  for (int y = 0; y < ctuH; y++)
  {
    for (int x = 0; x < ctuW; x+=8)
    {
      src8 = vld1_u8(src+x);
      src16  = vreinterpretq_s16_u16(vmovl_u8(src8));

      src2 = vshrq_n_s16(src16, 3);

      mask0 = vceqq_s16 (src2, r0);
      mask1 = vceqq_s16 (src2, r1);
      mask2 = vceqq_s16 (src2, r2);
      mask3 = vceqq_s16 (src2, r3);

      x0 = vbslq_s16(mask0, sao0, zeros);
      x1 = vbslq_s16(mask1, sao1, zeros);
      x2 = vbslq_s16(mask2, sao2, zeros);
      x3 = vbslq_s16(mask3, sao3, zeros);

      x0 = vorrq_s16(x0, x1);
      x2 = vorrq_s16(x2, x3);
      x0 = vorrq_s16(x0, x2);
      
      src16 = vaddq_s16(src16, x0);
      src8 = vqmovun_s16(src16);

      vst1_u8(dst+x, src8);
    }
    src += stride_src;
    dst += stride_dst;
  }
}


void ff_hevc_sao_edge_filter_8_neon(uint8_t* out_ptr, int out_stride, uint8_t* in_ptr , int in_stride, int SaoEoClass, int8_t* saoOffsetVal, int* edges, int ctuW, int ctuH, const int maxPixelValue){
  const int8_t pos[4][2][2] = {{ {-1, 0}, { 1, 0} },{ { 0,-1}, { 0, 1} }, { {-1,-1}, { 1, 1} }, { { 1,-1}, {-1, 1} }, };
  int init_y = 0, width = ctuW, height = ctuH;
  uint8_t* dst = out_ptr;
  uint8_t* src = in_ptr;
  int stride_dst = out_stride, stride_src = in_stride;

  if (SaoEoClass != 0) {
    int16x8_t x0, x1;
    uint8x8_t src8;
    if (edges[1]) {
      x1 = vdupq_n_s16(saoOffsetVal[2]);
      for (int x = 0; x < width; x += 8) {
        src8 = vld1_u8(src + x);
        x0  = vreinterpretq_s16_u16(vmovl_u8(src8));
        x0 = vaddq_s16(x0, x1);
        src8 = vqmovun_s16(x0);
        vst1_u8(dst+x, src8);
      };
      init_y = 1;
    }
    if (edges[3]) {
      int y_stride_dst = stride_dst * (ctuH - 1);
      int y_stride_src = stride_src * (ctuH - 1);
      x1 = vdupq_n_s16(saoOffsetVal[2]);
      for (int x = 0; x < width; x += 8) {
        src8 = vld1_u8(src + x + y_stride_src);
        x0  = vreinterpretq_s16_u16(vmovl_u8(src8));
        x0 = vaddq_s16(x0, x1);
        src8 = vqmovun_s16(x0);
        vst1_u8(dst + x + y_stride_dst, src8);
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
    
    int8_t sao_offset_val[8] = {saoOffsetVal[0], saoOffsetVal[1], saoOffsetVal[2], 
                                saoOffsetVal[3], saoOffsetVal[4], 0, 0, 0};
    int8x8_t offset = vld1_s8(sao_offset_val);

    for (int y = init_y; y < height; y++) {
      for (int x = 0; x < width; x += 8) {
        uint8x8_t x0 = vld1_u8(src + x + y_stride_src);
        uint8x8_t cmp0 = vld1_u8(src + x + y_stride_0_1);
        uint8x8_t cmp1 = vld1_u8(src + x + y_stride_1_1);

        uint8x8_t r2 = vmin_u8(x0, cmp0);
        uint8x8_t x1 = vceq_u8(cmp0, r2);
        uint8x8_t x2 = vceq_u8(x0, r2);
        int8x8_t diff0 = vsub_s8(vreinterpret_s8_u8(x2), vreinterpret_s8_u8(x1));

        r2 = vmin_u8(x0, cmp1);
        uint8x8_t x3 = vceq_u8(cmp1, r2);
        x2 = vceq_u8(x0, r2);
        int8x8_t diff1 = vsub_s8(vreinterpret_s8_u8(x2), vreinterpret_s8_u8(x3));

        diff0 = vadd_s8(diff0, diff1);
        int8x8_t index = vadd_s8(diff0, vdup_n_s8(2));

        uint8x8_t mask = vclt_u8(vreinterpret_u8_s8(index), vdup_n_u8(0x80));
        uint8x8_t index1 = vand_u8(vreinterpret_u8_s8(index), vdup_n_u8(15));
        index1 = vtbl1_u8(vreinterpret_u8_s8(offset), index1);
        int8x8_t offset1 = vreinterpret_s8_u8(vand_u8(index1, mask));

        int16x8_t src16  = vreinterpretq_s16_u16(vmovl_u8(x0));
        int16x8_t result = vaddw_s8(src16,offset1);

        //int16x8_t result = vaddl_s8(vreinterpret_s8_u8(x0),offset);
        x0 = vqmovun_s16(result);
        vst1_u8(dst + x + y_stride_dst, x0);
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

