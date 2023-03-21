/*
 * H.265 video codec.
 * ASM speedup module
 * mingyuan.myy@alibaba-inc.com
 */

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "x86.h"
// #include "x86/sse-motion.h"
#include "x86_idct.h"
#include "x86_sao.h"
#include "x86_dbk.h"
#include "x86_intrapred.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef __GNUC__
#include <cpuid.h>
#endif

void init_acceleration_functions_sse(struct acceleration_functions* accel)
{
  uint32_t ecx=0,edx=0;

#ifdef _MSC_VER
  uint32_t regs[4];
  int a = 1;

  __cpuid((int *)regs, (int)a);

  ecx = regs[2];
  edx = regs[3];
#else
  uint32_t eax,ebx;
  __get_cpuid(1, &eax,&ebx,&ecx,&edx);
#endif
  


  // printf("CPUID EAX=1 -> ECX=%x EDX=%x\n", regs[2], regs[3]);

  //int have_MMX    = !!(edx & (1<<23));
  int have_SSE    = !!(edx & (1<<25));
  int have_SSE4_1 = !!(ecx & (1<<19));

  // printf("MMX:%d SSE:%d SSE4_1:%d\n",have_MMX,have_SSE,have_SSE4_1);

  if (have_SSE) {
  }

#ifdef HAVE_SSE4_1
  //std::cout << "SSE4 is detected!" << std::endl;
#endif

#ifdef HAVE_AVX2
  //std::cout << "AVX2 is detected!" << std::endl;
#endif

#ifdef HAVE_AVX512
  //std::cout << "AVX512 is detected!" << std::endl;
#endif

#if HAVE_SSE4_1
  if (have_SSE4_1) {

    /*inverse transform*/
    accel->transform_4x4_dst_add_8 = ff_hevc_transform_4x4_luma_add_8_sse4;
    accel->transform_add_8[0] = ff_hevc_transform_4x4_add_8_sse4;
    accel->transform_add_8[1] = ff_hevc_transform_8x8_add_8_sse4;
    accel->transform_add_8[2] = ff_hevc_transform_16x16_add_8_sse4;
    accel->transform_add_8[3] = ff_hevc_transform_32x32_add_8_sse4;

    accel->transform_dc_add_8[0] = ff_hevc_transform_4x4_dc_add_8_sse4;
    accel->transform_dc_add_8[1] = ff_hevc_transform_8x8_dc_add_8_sse4;
    accel->transform_dc_add_8[2] = ff_hevc_transform_16x16_dc_add_8_sse4;
    accel->transform_dc_add_8[3] = ff_hevc_transform_32x32_dc_add_8_sse4;

    accel->transform_skip_residual16 = ff_hevc_transform_skip_residual16_sse4;
    accel->add_residual16_8  = ff_hevc_residual16_add_8_sse4;

    /*SAO filtering*/
    accel->sao_band_filter_8 = ff_hevc_sao_band_filter_8_sse4;
    accel->sao_edge_filter_8 = ff_hevc_sao_edge_filter_8_sse4;


    /*deblocking filter*/
    accel->loop_filter_luma_8 = ff_hevc_loop_filter_luma_8_sse4;
    accel->loop_filter_chroma_8 = ff_hevc_loop_filter_chroma_8_sse4;

    // /*intra prediction*/
    accel->intra_pred_dc_8[0]  = intra_prediction_DC_4_8_sse4 ; 
    accel->intra_pred_dc_8[1]  = intra_prediction_DC_8_8_sse4 ;
    accel->intra_pred_dc_8[2]  = intra_prediction_DC_16_8_sse4 ;
    accel->intra_pred_dc_8[3]  = intra_prediction_DC_32_8_sse4 ;

    accel->intra_prediction_angular_8[0]  = intra_prediction_angular_2_9_sse4;
    accel->intra_prediction_angular_8[1]  = intra_prediction_angular_10_17_sse4;
    accel->intra_prediction_angular_8[2]  = intra_prediction_angular_18_26_sse4;
    accel->intra_prediction_angular_8[3]  = intra_prediction_angular_27_34_sse4;
    
    accel->intra_prediction_planar_8    = intra_prediction_planar_8_sse4 ;

    accel->intra_prediction_sample_filtering_8 = intra_prediction_sample_filtering_sse4;


    // accel->put_unweighted_pred_8   = ff_hevc_put_unweighted_pred_8_sse;
    // accel->put_weighted_pred_avg_8 = ff_hevc_put_weighted_pred_avg_8_sse;

    // accel->put_hevc_epel_8    = ff_hevc_put_hevc_epel_pixels_8_sse;
    // accel->put_hevc_epel_h_8  = ff_hevc_put_hevc_epel_h_8_sse;
    // accel->put_hevc_epel_v_8  = ff_hevc_put_hevc_epel_v_8_sse;
    // accel->put_hevc_epel_hv_8 = ff_hevc_put_hevc_epel_hv_8_sse;

    // accel->put_hevc_qpel_8[0][0] = ff_hevc_put_hevc_qpel_pixels_8_sse;
    // accel->put_hevc_qpel_8[0][1] = ff_hevc_put_hevc_qpel_v_1_8_sse;
    // accel->put_hevc_qpel_8[0][2] = ff_hevc_put_hevc_qpel_v_2_8_sse;
    // accel->put_hevc_qpel_8[0][3] = ff_hevc_put_hevc_qpel_v_3_8_sse;
    // accel->put_hevc_qpel_8[1][0] = ff_hevc_put_hevc_qpel_h_1_8_sse;
    // accel->put_hevc_qpel_8[1][1] = ff_hevc_put_hevc_qpel_h_1_v_1_sse;
    // accel->put_hevc_qpel_8[1][2] = ff_hevc_put_hevc_qpel_h_1_v_2_sse;
    // accel->put_hevc_qpel_8[1][3] = ff_hevc_put_hevc_qpel_h_1_v_3_sse;
    // accel->put_hevc_qpel_8[2][0] = ff_hevc_put_hevc_qpel_h_2_8_sse;
    // accel->put_hevc_qpel_8[2][1] = ff_hevc_put_hevc_qpel_h_2_v_1_sse;
    // accel->put_hevc_qpel_8[2][2] = ff_hevc_put_hevc_qpel_h_2_v_2_sse;
    // accel->put_hevc_qpel_8[2][3] = ff_hevc_put_hevc_qpel_h_2_v_3_sse;
    // accel->put_hevc_qpel_8[3][0] = ff_hevc_put_hevc_qpel_h_3_8_sse;
    // accel->put_hevc_qpel_8[3][1] = ff_hevc_put_hevc_qpel_h_3_v_1_sse;
    // accel->put_hevc_qpel_8[3][2] = ff_hevc_put_hevc_qpel_h_3_v_2_sse;
    // accel->put_hevc_qpel_8[3][3] = ff_hevc_put_hevc_qpel_h_3_v_3_sse;

    // accel->transform_skip_8 = ff_hevc_transform_skip_8_sse;

    // actually, for these two functions, the scalar fallback seems to be faster than the SSE code
    //accel->transform_4x4_luma_add_8 = ff_hevc_transform_4x4_luma_add_8_sse4; // SSE-4 only TODO
    //accel->transform_4x4_add_8   = ff_hevc_transform_4x4_add_8_sse4;

//    accel->transform_add_8[1] = ff_hevc_transform_8x8_add_8_sse4;
//    accel->transform_add_8[2] = ff_hevc_transform_16x16_add_8_sse4;
//    accel->transform_add_8[3] = ff_hevc_transform_32x32_add_8_sse4;
  }
#endif

#if HAVE_AVX2
  /*inverse transform*/
  accel->transform_add_8[2] = ff_hevc_transform_16x16_add_8_avx2;
  accel->transform_add_8[3] = ff_hevc_transform_32x32_add_8_avx2;
  accel->transform_dc_add_8[2] = ff_hevc_transform_16x16_dc_add_8_avx2;
  accel->transform_dc_add_8[3] = ff_hevc_transform_32x32_dc_add_8_avx2;
  accel->transform_skip_residual16 = ff_hevc_transform_skip_residual16_avx2;

  /*SAO filtering*/
  accel->sao_band_filter_8 = ff_hevc_sao_band_filter_8_avx2;
  accel->sao_edge_filter_8 = ff_hevc_sao_edge_filter_8_avx2;


  accel->intra_pred_dc_8[3]  = intra_prediction_DC_32_8_avx2;

  accel->intra_prediction_planar_8    = intra_prediction_planar_8_avx2 ;
#endif
}
