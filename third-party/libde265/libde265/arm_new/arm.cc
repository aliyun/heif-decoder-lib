#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include "arm.h"
#include "neon_common.h"
#include "neon_dct.h"
#include "neon_intrapred.h"
#include "neon_dbk.h"



void init_acceleration_functions_arm(struct acceleration_functions* accel)
{
#ifdef HAVE_ARM
  //std::cout << "arm is detected!" << std::endl;
#endif
#ifdef HAVE_ARM64
  //std::cout << "arm64 is detected!" << std::endl;
#endif
  

  /*inverse transform*/
  accel->transform_4x4_dst_add_8 = ff_hevc_transform_4x4_luma_add_8_neon;

  accel->transform_dc_add_8[0] = ff_hevc_transform_4x4_dc_add_8_neon;
  accel->transform_dc_add_8[1] = ff_hevc_transform_8x8_dc_add_8_neon;
  accel->transform_dc_add_8[2] = ff_hevc_transform_16x16_dc_add_8_neon;
  accel->transform_dc_add_8[3] = ff_hevc_transform_32x32_dc_add_8_neon;

  accel->transform_add_8[0] = ff_hevc_transform_4x4_add_8_neon;
  accel->transform_add_8[1] = ff_hevc_transform_8x8_add_8_neon;
  accel->transform_add_8[2] = ff_hevc_transform_16x16_add_8_neon;
  accel->transform_add_8[3] = ff_hevc_transform_32x32_add_8_neon;
  accel->transform_skip_residual16 = ff_hevc_transform_skip_residual16;
  accel->add_residual16_8  = ff_hevc_residual16_add_8_neon;

  accel->intra_pred_dc_8[0]  = intra_prediction_DC_neon_8 ; 
  accel->intra_pred_dc_8[1]  = intra_prediction_DC_neon_8 ;
  accel->intra_pred_dc_8[2]  = intra_prediction_DC_neon_8 ;
  accel->intra_pred_dc_8[3]  = intra_prediction_DC_neon_8 ;
//  accel->intra_pred_dc_16 = intra_prediction_DC_neon_16 ;
  accel->intra_prediction_angular_8[0]  = intra_prediction_angular_2_9_neon ;
  accel->intra_prediction_angular_8[1]  = intra_prediction_angular_10_17_neon;
  accel->intra_prediction_angular_8[2]  = intra_prediction_angular_18_26_neon;
  accel->intra_prediction_angular_8[3]  = intra_prediction_angular_27_34_neon;
//  accel->intra_prediction_angular_16[0] = intra_prediction_angular_2_9_neon ;
//  accel->intra_prediction_angular_16[1] = intra_prediction_angular_10_17_neon;
//  accel->intra_prediction_angular_16[2] = intra_prediction_angular_18_26_neon;
//  accel->intra_prediction_angular_16[3] = intra_prediction_angular_27_34_neon;

  accel->intra_prediction_sample_filtering_8 = intra_prediction_sample_filtering_neon;
//  accel->intra_prediction_sample_filtering_16= intra_prediction_sample_filtering_neon ;

  accel->intra_prediction_planar_8    = intra_prediction_planar_neon ;
//  accel->intra_prediction_planar_16   = intra_prediction_planar_neon ;


  /*deblocking filter*/
  accel->loop_filter_luma_8 = ff_hevc_loop_filter_luma_8_neon;
  accel->loop_filter_chroma_8 = ff_hevc_loop_filter_chroma_8_neon;

  /*SAO filtering*/
  accel->sao_band_filter_8 = ff_hevc_sao_band_filter_8_neon;
  accel->sao_edge_filter_8 = ff_hevc_sao_edge_filter_8_neon;
}
