/*
 * H.265 video codec.
 * Copyright (c) 2013-2014 struktur AG, Dirk Farin <farin@struktur.de>
 *
 * This file is part of libde265.
 *
 * libde265 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * libde265 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libde265.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DE265_ACCELERATION_H
#define DE265_ACCELERATION_H

#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include "slice.h"

struct acceleration_functions
{
  void (*put_weighted_pred_avg_8)(uint8_t *_dst, ptrdiff_t dststride,
                                  const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                                  int width, int height);

  void (*put_unweighted_pred_8)(uint8_t *_dst, ptrdiff_t dststride,
                                const int16_t *src, ptrdiff_t srcstride,
                                int width, int height);

  void (*put_weighted_pred_8)(uint8_t *_dst, ptrdiff_t dststride,
                              const int16_t *src, ptrdiff_t srcstride,
                              int width, int height,
                              int w,int o,int log2WD);
  void (*put_weighted_bipred_8)(uint8_t *_dst, ptrdiff_t dststride,
                                const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                                int width, int height,
                                int w1,int o1, int w2,int o2, int log2WD);


  void (*put_weighted_pred_avg_16)(uint16_t *_dst, ptrdiff_t dststride,
                                  const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                                   int width, int height, int bit_depth);

  void (*put_unweighted_pred_16)(uint16_t *_dst, ptrdiff_t dststride,
                                const int16_t *src, ptrdiff_t srcstride,
                                int width, int height, int bit_depth);

  void (*put_weighted_pred_16)(uint16_t *_dst, ptrdiff_t dststride,
                              const int16_t *src, ptrdiff_t srcstride,
                              int width, int height,
                              int w,int o,int log2WD, int bit_depth);
  void (*put_weighted_bipred_16)(uint16_t *_dst, ptrdiff_t dststride,
                                const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                                int width, int height,
                                int w1,int o1, int w2,int o2, int log2WD, int bit_depth);


  void put_weighted_pred_avg(void *_dst, ptrdiff_t dststride,
                             const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                             int width, int height, int bit_depth) const;

  void put_unweighted_pred(void *_dst, ptrdiff_t dststride,
                           const int16_t *src, ptrdiff_t srcstride,
                           int width, int height, int bit_depth) const;

  void put_weighted_pred(void *_dst, ptrdiff_t dststride,
                         const int16_t *src, ptrdiff_t srcstride,
                         int width, int height,
                         int w,int o,int log2WD, int bit_depth) const;
  void put_weighted_bipred(void *_dst, ptrdiff_t dststride,
                           const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                           int width, int height,
                           int w1,int o1, int w2,int o2, int log2WD, int bit_depth) const;




  void (*put_hevc_epel_8)(int16_t *dst, ptrdiff_t dststride,
                          const uint8_t *src, ptrdiff_t srcstride, int width, int height,
                          int mx, int my, int16_t* mcbuffer);
  void (*put_hevc_epel_h_8)(int16_t *dst, ptrdiff_t dststride,
                            const uint8_t *src, ptrdiff_t srcstride, int width, int height,
                            int mx, int my, int16_t* mcbuffer, int bit_depth);
  void (*put_hevc_epel_v_8)(int16_t *dst, ptrdiff_t dststride,
                            const uint8_t *src, ptrdiff_t srcstride, int width, int height,
                            int mx, int my, int16_t* mcbuffer, int bit_depth);
  void (*put_hevc_epel_hv_8)(int16_t *dst, ptrdiff_t dststride,
                             const uint8_t *src, ptrdiff_t srcstride, int width, int height,
                             int mx, int my, int16_t* mcbuffer, int bit_depth);

  void (*put_hevc_qpel_8[4][4])(int16_t *dst, ptrdiff_t dststride,
                                const uint8_t *src, ptrdiff_t srcstride, int width, int height,
                                int16_t* mcbuffer);


  void (*put_hevc_epel_16)(int16_t *dst, ptrdiff_t dststride,
                           const uint16_t *src, ptrdiff_t srcstride, int width, int height,
                           int mx, int my, int16_t* mcbuffer, int bit_depth);
  void (*put_hevc_epel_h_16)(int16_t *dst, ptrdiff_t dststride,
                             const uint16_t *src, ptrdiff_t srcstride, int width, int height,
                            int mx, int my, int16_t* mcbuffer, int bit_depth);
  void (*put_hevc_epel_v_16)(int16_t *dst, ptrdiff_t dststride,
                             const uint16_t *src, ptrdiff_t srcstride, int width, int height,
                             int mx, int my, int16_t* mcbuffer, int bit_depth);
  void (*put_hevc_epel_hv_16)(int16_t *dst, ptrdiff_t dststride,
                              const uint16_t *src, ptrdiff_t srcstride, int width, int height,
                              int mx, int my, int16_t* mcbuffer, int bit_depth);

  void (*put_hevc_qpel_16[4][4])(int16_t *dst, ptrdiff_t dststride,
                                 const uint16_t *src, ptrdiff_t srcstride, int width, int height,
                                 int16_t* mcbuffer, int bit_depth);


  void put_hevc_epel(int16_t *dst, ptrdiff_t dststride,
                     const void *src, ptrdiff_t srcstride, int width, int height,
                     int mx, int my, int16_t* mcbuffer, int bit_depth) const;
  void put_hevc_epel_h(int16_t *dst, ptrdiff_t dststride,
                       const void *src, ptrdiff_t srcstride, int width, int height,
                       int mx, int my, int16_t* mcbuffer, int bit_depth) const;
  void put_hevc_epel_v(int16_t *dst, ptrdiff_t dststride,
                       const void *src, ptrdiff_t srcstride, int width, int height,
                       int mx, int my, int16_t* mcbuffer, int bit_depth) const;
  void put_hevc_epel_hv(int16_t *dst, ptrdiff_t dststride,
                        const void *src, ptrdiff_t srcstride, int width, int height,
                        int mx, int my, int16_t* mcbuffer, int bit_depth) const;

  void put_hevc_qpel(int16_t *dst, ptrdiff_t dststride,
                     const void *src, ptrdiff_t srcstride, int width, int height,
                     int16_t* mcbuffer, int dX,int dY, int bit_depth) const;


  // --- inverse transforms ---

  void (*transform_bypass)(int32_t *residual, const int16_t *coeffs, int nT);
  void (*transform_bypass_rdpcm_v)(int32_t *r, const int16_t *coeffs, int nT);
  void (*transform_bypass_rdpcm_h)(int32_t *r, const int16_t *coeffs, int nT);

  // 8 bit

  void (*transform_skip_8)(uint8_t *_dst, const int16_t *coeffs, ptrdiff_t _stride); // no transform
  void (*transform_skip_rdpcm_v_8)(uint8_t *_dst, const int16_t *coeffs, int nT, ptrdiff_t _stride);
  void (*transform_skip_rdpcm_h_8)(uint8_t *_dst, const int16_t *coeffs, int nT, ptrdiff_t _stride);
  void (*transform_4x4_dst_add_8)(uint8_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth); // iDST
  void (*transform_add_8[4])(uint8_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int16_t col_limit); // iDCT
  void (*transform_dc_add_8[4])(uint8_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int16_t col_limit); // iDCT_DC

  // 9-16 bit

  void (*transform_skip_16)(uint16_t *_dst, const int16_t *coeffs, ptrdiff_t _stride, int bit_depth); // no transform
  void (*transform_4x4_dst_add_16)(uint16_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth); // iDST
  void (*transform_add_16[4])(uint16_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth, int16_t col_limit); // iDCT
  void (*transform_dc_add_16[4])(uint16_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth, int16_t col_limit); // iDCT_DC


  void (*rotate_coefficients)(int16_t *coeff, int nT);

  void (*transform_idst_4x4)(int32_t *dst, const int16_t *coeffs, int bdShift, int max_coeff_bits);
  void (*transform_idct_4x4)(int32_t *dst, const int16_t *coeffs, int bdShift, int max_coeff_bits);
  void (*transform_idct_8x8)(int32_t *dst, const int16_t *coeffs, int bdShift, int max_coeff_bits);
  void (*transform_idct_16x16)(int32_t *dst,const int16_t *coeffs,int bdShift, int max_coeff_bits);
  void (*transform_idct_32x32)(int32_t *dst,const int16_t *coeffs,int bdShift, int max_coeff_bits);
  void (*add_residual_8)(uint8_t *dst, ptrdiff_t stride, const int32_t* r, int nT, int bit_depth);
  void (*add_residual_16)(uint16_t *dst,ptrdiff_t stride,const int32_t* r, int nT, int bit_depth);
  void (*add_residual16_8)(uint8_t *dst, ptrdiff_t stride, const int16_t* r, int nT, int bit_depth);
  void (*add_residual16_16)(uint16_t *dst,ptrdiff_t stride,const int16_t* r, int nT, int bit_depth);

  void (*loop_filter_luma_8)(uint8_t *dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth);
  void (*loop_filter_luma_16)(uint16_t *dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth);
  void (*loop_filter_luma_c_8)(uint8_t *dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth);
  void (*loop_filter_luma_c_16)(uint16_t *dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth);

#ifdef OPT_4
  void (*loop_filter_chroma_8)(uint8_t *dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth);
  void (*loop_filter_chroma_16)(uint16_t *dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth);
  void (*loop_filter_chroma_c_8)(uint8_t *dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth);
  void (*loop_filter_chroma_c_16)(uint16_t *dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth);
#else
  void (*loop_filter_chroma_8)(uint8_t *dst, bool vertical, ptrdiff_t stride, int tc, bool filterP, bool filterQ, int bitDepth);
  void (*loop_filter_chroma_16)(uint16_t *dst, bool vertical, ptrdiff_t stride, int tc, bool filterP, bool filterQ, int bitDepth);
  void (*loop_filter_chroma_c_8)(uint8_t *dst, bool vertical, ptrdiff_t stride, int tc, bool filterP, bool filterQ, int bitDepth);
  void (*loop_filter_chroma_c_16)(uint16_t *dst, bool vertical, ptrdiff_t stride,  int tc, bool filterP, bool filterQ, int bitDepth);
#endif


  void (*sao_band_filter_8)(uint8_t *_dst, int stride_dst, /*const*/ uint8_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, int saoLeftClass, const int8_t *saoOffsetVal, int bandshift, const int maxPixelValue) ;
  void (*sao_band_filter_16)(uint16_t *_dst, int stride_dst, /*const*/ uint16_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, int saoLeftClass, const int8_t *saoOffsetVal, int bandshift, const int maxPixelValue) ;
  template <class pixel_t>
  void sao_band_filter(pixel_t *_dst, int stride_dst, /*const*/ pixel_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, int saoLeftClass, const int8_t *saoOffsetVal, int bandshift, const int maxPixelValue) const;


  void (*sao_edge_filter_8)(uint8_t* _dst, int _stride_dst, uint8_t* _src , int _stride_src, int SaoEoClass, int8_t* saoOffsetVal, int* edges, int ctuW, int ctuH, const int maxPixelValue);
  void (*sao_edge_filter_16)(uint16_t* _dst, int _stride_dst, uint16_t* _src , int _stride_src, int SaoEoClass, int8_t* saoOffsetVal, int* edges, int ctuW, int ctuH, const int maxPixelValue);
  template <class pixel_t>
  void sao_edge_filter(pixel_t* _dst, int _stride_dst, pixel_t* _src , int _stride_src, int SaoEoClass, int8_t* saoOffsetVal, int* edges, int ctuW, int ctuH, const int maxPixelValue) const;
 
  template <class pixel_t>
  void loop_filter_luma(pixel_t *dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const;
  template <class pixel_t>
  void loop_filter_luma_c(pixel_t *dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const;

#ifdef OPT_4
  template <class pixel_t>
  void loop_filter_chroma(pixel_t *dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const;
  template <class pixel_t>
  void loop_filter_chroma_c(pixel_t *dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q,  int bitDepth) const;
#else
  template <class pixel_t>
  void loop_filter_chroma(pixel_t *dst, bool vertical, ptrdiff_t stride, int tc, bool filterP, bool filterQ, int bitDepth) const;
  template <class pixel_t>
  void loop_filter_chroma_c(pixel_t *dst, bool vertical, ptrdiff_t stride, int tc, bool filterP, bool filterQ, int bitDepth) const;
#endif

  template <class pixel_t>
  void add_residual(pixel_t *dst, ptrdiff_t stride, const int32_t* r, int nT, int bit_depth) const;
  template <class pixel_t>
  void add_residual16(pixel_t *dst, ptrdiff_t stride, const int16_t* r, int nT, int bit_depth) const;

  void (*rdpcm_v)(int32_t* residual, const int16_t* coeffs, int nT,int tsShift,int bdShift);
  void (*rdpcm_h)(int32_t* residual, const int16_t* coeffs, int nT,int tsShift,int bdShift);
  void (*rdpcm_v16)(int16_t* residual, const int16_t* coeffs, int nT,int tsShift,int bdShift);
  void (*rdpcm_h16)(int16_t* residual, const int16_t* coeffs, int nT,int tsShift,int bdShift);

  void (*transform_skip_residual)(int32_t *residual, const int16_t *coeffs, int nT,
                                  int tsShift,int bdShift);
  void (*transform_skip_residual16)(int16_t *residual, const int16_t *coeffs, int nT,
                                  int tsShift,int bdShift);


  template <class pixel_t> void transform_skip(pixel_t *dst, const int16_t *coeffs, ptrdiff_t stride, int bit_depth) const;
  template <class pixel_t> void transform_skip_rdpcm_v(pixel_t *dst, const int16_t *coeffs, int nT, ptrdiff_t stride, int bit_depth) const;
  template <class pixel_t> void transform_skip_rdpcm_h(pixel_t *dst, const int16_t *coeffs, int nT, ptrdiff_t stride, int bit_depth) const;
  template <class pixel_t> void transform_4x4_dst_add(pixel_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth) const;
  template <class pixel_t> void transform_add(int sizeIdx, pixel_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth, int16_t col_limit) const;
  template <class pixel_t> void transform_dc_add(int sizeIdx, pixel_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth, int16_t col_limit) const;


  // --- forward transforms ---

  void (*fwd_transform_4x4_dst_8)(int16_t *coeffs, const int16_t* src, ptrdiff_t stride); // fDST

  // indexed with (log2TbSize-2)
  void (*fwd_transform_8[4])     (int16_t *coeffs, const int16_t *src, ptrdiff_t stride); // fDCT


  // forward Hadamard transform (without scaling factor)
  // (4x4,8x8,16x16,32x32) indexed with (log2TbSize-2)
  void (*hadamard_transform_8[4])     (int16_t *coeffs, const int16_t *src, ptrdiff_t stride);

  // ---- intra prediction ----

  void (*intra_pred_dc_8[4])(uint8_t *dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border);
  // To do, for 10 bitdepth,
  void (*intra_pred_dc_16[4])(uint16_t *dst, int dstStride, int nT, int cIdx, /* const */ uint16_t *border); 

  template <class pixel_t> void intra_pred_dc(pixel_t *dst, int dstStride, int nT, int log2TrafoSize, int cIdx, /* const */ pixel_t *border) const ;

  void (*intra_prediction_angular_8[4])(uint8_t *dst, int dstStride, int bit_depth, bool disableIntraBoundaryFilter, int xB0,int yB0, \
                                        enum IntraPredMode intraPredMode, int nT,int cIdx, uint8_t * border);
  void (*intra_prediction_angular_16[4])(uint16_t *dst, int dstStride, int bit_depth, bool disableIntraBoundaryFilter, int xB0,int yB0,  \
                                         enum IntraPredMode intraPredMode, int nT,int cIdx, uint16_t * border);
  template <class pixel_t> void intra_prediction_angular(int AngIdx, pixel_t *dst, int dstStride, int bit_depth, bool disableIntraBoundaryFilter, int xB0,int yB0, \
                                                         enum IntraPredMode intraPredMode, int nT,int cIdx, pixel_t *border) const  ;

  void (*intra_prediction_sample_filtering_8 )(const seq_parameter_set& sps, uint8_t  * p, int nT, int cIdx, enum IntraPredMode intraPredMode) ;
  void (*intra_prediction_sample_filtering_16)(const seq_parameter_set& sps, uint16_t * p, int nT, int cIdx, enum IntraPredMode intraPredMode) ;
  template <class pixel_t> void intra_prediction_sample_filtering(const seq_parameter_set& sps, pixel_t* p, int nT, int cIdx, enum IntraPredMode intraPredMode) const ;

  void (*intra_prediction_planar_8)(uint8_t  *dst, int dstStride, int nT,int cIdx, uint8_t  *border);
  void (*intra_prediction_planar_16)(uint16_t *dst, int dstStride, int nT,int cIdx, uint16_t *border);
  template <class pixel_t> void intra_prediction_planar(pixel_t *dst, int dstStride, int nT, int log2TrafoSize, int cIdx, pixel_t *border) const ;

};


/*
template <> inline void acceleration_functions::put_weighted_pred_avg<uint8_t>(uint8_t *_dst, ptrdiff_t dststride,
                                                                               const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                                                                               int width, int height, int bit_depth) { put_weighted_pred_avg_8(_dst,dststride,src1,src2,srcstride,width,height); }
template <> inline void acceleration_functions::put_weighted_pred_avg<uint16_t>(uint16_t *_dst, ptrdiff_t dststride,
                                                                                const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                                                                                int width, int height, int bit_depth) { put_weighted_pred_avg_16(_dst,dststride,src1,src2,
                                                                                                                                                 srcstride,width,height,bit_depth); }

template <> inline void acceleration_functions::put_unweighted_pred<uint8_t>(uint8_t *_dst, ptrdiff_t dststride,
                                                                             const int16_t *src, ptrdiff_t srcstride,
                                                                             int width, int height, int bit_depth) { put_unweighted_pred_8(_dst,dststride,src,srcstride,width,height); }
template <> inline void acceleration_functions::put_unweighted_pred<uint16_t>(uint16_t *_dst, ptrdiff_t dststride,
                                                                              const int16_t *src, ptrdiff_t srcstride,
                                                                              int width, int height, int bit_depth) { put_unweighted_pred_16(_dst,dststride,src,srcstride,width,height,bit_depth); }

template <> inline void acceleration_functions::put_weighted_pred<uint8_t>(uint8_t *_dst, ptrdiff_t dststride,
                                                                           const int16_t *src, ptrdiff_t srcstride,
                                                                           int width, int height,
                                                                           int w,int o,int log2WD, int bit_depth) { put_weighted_pred_8(_dst,dststride,src,srcstride,width,height,w,o,log2WD); }
template <> inline void acceleration_functions::put_weighted_pred<uint16_t>(uint16_t *_dst, ptrdiff_t dststride,
                                                                            const int16_t *src, ptrdiff_t srcstride,
                                                                            int width, int height,
                                                                            int w,int o,int log2WD, int bit_depth) { put_weighted_pred_16(_dst,dststride,src,srcstride,width,height,w,o,log2WD,bit_depth); }

template <> inline void acceleration_functions::put_weighted_bipred<uint8_t>(uint8_t *_dst, ptrdiff_t dststride,
                                                                             const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                                                                             int width, int height,
                                                                             int w1,int o1, int w2,int o2, int log2WD, int bit_depth) { put_weighted_bipred_8(_dst,dststride,src1,src2,srcstride,
                                                                                                                                                              width,height,
                                                                                                                                                              w1,o1,w2,o2,log2WD); }
template <> inline void acceleration_functions::put_weighted_bipred<uint16_t>(uint16_t *_dst, ptrdiff_t dststride,
                                                                              const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                                                                              int width, int height,
                                                                              int w1,int o1, int w2,int o2, int log2WD, int bit_depth) { put_weighted_bipred_16(_dst,dststride,src1,src2,srcstride,
                                                                                                                                                                width,height,
                                                                                                                                                                w1,o1,w2,o2,log2WD,bit_depth); }
*/


inline void acceleration_functions::put_weighted_pred_avg(void* _dst, ptrdiff_t dststride,
                                                          const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                                                          int width, int height, int bit_depth) const
{
  if (bit_depth <= 8)
    put_weighted_pred_avg_8((uint8_t*)_dst,dststride,src1,src2,srcstride,width,height);
  else
    put_weighted_pred_avg_16((uint16_t*)_dst,dststride,src1,src2,srcstride,width,height,bit_depth);
}


inline void acceleration_functions::put_unweighted_pred(void* _dst, ptrdiff_t dststride,
                                                        const int16_t *src, ptrdiff_t srcstride,
                                                        int width, int height, int bit_depth) const
{
  if (bit_depth <= 8)
    put_unweighted_pred_8((uint8_t*)_dst,dststride,src,srcstride,width,height);
  else
    put_unweighted_pred_16((uint16_t*)_dst,dststride,src,srcstride,width,height,bit_depth);
}


inline void acceleration_functions::put_weighted_pred(void* _dst, ptrdiff_t dststride,
                                                      const int16_t *src, ptrdiff_t srcstride,
                                                      int width, int height,
                                                      int w,int o,int log2WD, int bit_depth) const
{
  if (bit_depth <= 8)
    put_weighted_pred_8((uint8_t*)_dst,dststride,src,srcstride,width,height,w,o,log2WD);
  else
    put_weighted_pred_16((uint16_t*)_dst,dststride,src,srcstride,width,height,w,o,log2WD,bit_depth);
}


inline void acceleration_functions::put_weighted_bipred(void* _dst, ptrdiff_t dststride,
                                                        const int16_t *src1, const int16_t *src2, ptrdiff_t srcstride,
                                                        int width, int height,
                                                        int w1,int o1, int w2,int o2, int log2WD, int bit_depth) const
{
  if (bit_depth <= 8)
    put_weighted_bipred_8((uint8_t*)_dst,dststride,src1,src2,srcstride, width,height, w1,o1,w2,o2,log2WD);
  else
    put_weighted_bipred_16((uint16_t*)_dst,dststride,src1,src2,srcstride, width,height, w1,o1,w2,o2,log2WD,bit_depth);
}



inline void acceleration_functions::put_hevc_epel(int16_t *dst, ptrdiff_t dststride,
                                                  const void *src, ptrdiff_t srcstride, int width, int height,
                                                  int mx, int my, int16_t* mcbuffer, int bit_depth) const
{
  if (bit_depth <= 8)
    put_hevc_epel_8(dst,dststride,(const uint8_t*)src,srcstride,width,height,mx,my,mcbuffer);
  else
    put_hevc_epel_16(dst,dststride,(const uint16_t*)src,srcstride,width,height,mx,my,mcbuffer, bit_depth);
}

inline void acceleration_functions::put_hevc_epel_h(int16_t *dst, ptrdiff_t dststride,
                                                    const void *src, ptrdiff_t srcstride, int width, int height,
                                                    int mx, int my, int16_t* mcbuffer, int bit_depth) const
{
  if (bit_depth <= 8)
    put_hevc_epel_h_8(dst,dststride,(const uint8_t*)src,srcstride,width,height,mx,my,mcbuffer,bit_depth);
  else
    put_hevc_epel_h_16(dst,dststride,(const uint16_t*)src,srcstride,width,height,mx,my,mcbuffer,bit_depth);
}

inline void acceleration_functions::put_hevc_epel_v(int16_t *dst, ptrdiff_t dststride,
                                                    const void *src, ptrdiff_t srcstride, int width, int height,
                                                    int mx, int my, int16_t* mcbuffer, int bit_depth) const
{
  if (bit_depth <= 8)
    put_hevc_epel_v_8(dst,dststride,(const uint8_t*)src,srcstride,width,height,mx,my,mcbuffer,bit_depth);
  else
    put_hevc_epel_v_16(dst,dststride,(const uint16_t*)src,srcstride,width,height,mx,my,mcbuffer, bit_depth);
}

inline void acceleration_functions::put_hevc_epel_hv(int16_t *dst, ptrdiff_t dststride,
                                                     const void *src, ptrdiff_t srcstride, int width, int height,
                                                     int mx, int my, int16_t* mcbuffer, int bit_depth) const
{
  if (bit_depth <= 8)
    put_hevc_epel_hv_8(dst,dststride,(const uint8_t*)src,srcstride,width,height,mx,my,mcbuffer,bit_depth);
  else
    put_hevc_epel_hv_16(dst,dststride,(const uint16_t*)src,srcstride,width,height,mx,my,mcbuffer, bit_depth);
}

inline void acceleration_functions::put_hevc_qpel(int16_t *dst, ptrdiff_t dststride,
                                                  const void *src, ptrdiff_t srcstride, int width, int height,
                                                  int16_t* mcbuffer, int dX,int dY, int bit_depth) const
{
  if (bit_depth <= 8)
    put_hevc_qpel_8[dX][dY](dst,dststride,(const uint8_t*)src,srcstride,width,height,mcbuffer);
  else
    put_hevc_qpel_16[dX][dY](dst,dststride,(const uint16_t*)src,srcstride,width,height,mcbuffer, bit_depth);
}

template <> inline void acceleration_functions::transform_skip<uint8_t>(uint8_t *dst, const int16_t *coeffs,ptrdiff_t stride, int bit_depth) const { transform_skip_8(dst,coeffs,stride); }
template <> inline void acceleration_functions::transform_skip<uint16_t>(uint16_t *dst, const int16_t *coeffs, ptrdiff_t stride, int bit_depth) const { transform_skip_16(dst,coeffs,stride, bit_depth); }

template <> inline void acceleration_functions::transform_skip_rdpcm_v<uint8_t>(uint8_t *dst, const int16_t *coeffs, int nT, ptrdiff_t stride, int bit_depth) const { assert(bit_depth==8); transform_skip_rdpcm_v_8(dst,coeffs,nT,stride); }
template <> inline void acceleration_functions::transform_skip_rdpcm_h<uint8_t>(uint8_t *dst, const int16_t *coeffs, int nT, ptrdiff_t stride, int bit_depth) const { assert(bit_depth==8); transform_skip_rdpcm_h_8(dst,coeffs,nT,stride); }
template <> inline void acceleration_functions::transform_skip_rdpcm_v<uint16_t>(uint16_t *dst, const int16_t *coeffs, int nT, ptrdiff_t stride, int bit_depth) const { assert(false); /*transform_skip_rdpcm_v_8(dst,coeffs,nT,stride);*/ }
template <> inline void acceleration_functions::transform_skip_rdpcm_h<uint16_t>(uint16_t *dst, const int16_t *coeffs, int nT, ptrdiff_t stride, int bit_depth) const { assert(false); /*transform_skip_rdpcm_h_8(dst,coeffs,nT,stride);*/ }


template <> inline void acceleration_functions::transform_4x4_dst_add<uint8_t>(uint8_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride,int bit_depth) const { transform_4x4_dst_add_8(dst,coeffs,stride, bit_depth); }
template <> inline void acceleration_functions::transform_4x4_dst_add<uint16_t>(uint16_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride,int bit_depth) const { transform_4x4_dst_add_16(dst,coeffs,stride,bit_depth); }

template <> inline void acceleration_functions::transform_add<uint8_t>(int sizeIdx, uint8_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth, int16_t col_limit) const { transform_add_8[sizeIdx](dst,coeffs,stride, col_limit); }
template <> inline void acceleration_functions::transform_add<uint16_t>(int sizeIdx, uint16_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth, int16_t col_limit) const { transform_add_16[sizeIdx](dst,coeffs,stride,bit_depth, col_limit); }

template <> inline void acceleration_functions::transform_dc_add<uint8_t>(int sizeIdx, uint8_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth, int16_t col_limit) const { transform_dc_add_8[sizeIdx](dst,coeffs,stride, col_limit); }
template <> inline void acceleration_functions::transform_dc_add<uint16_t>(int sizeIdx, uint16_t *dst, /*const*/ int16_t *coeffs, ptrdiff_t stride, int bit_depth, int16_t col_limit) const { transform_dc_add_16[sizeIdx](dst,coeffs,stride,bit_depth, col_limit); }

template <> inline void acceleration_functions::add_residual(uint8_t *dst,  ptrdiff_t stride, const int32_t* r, int nT, int bit_depth) const { add_residual_8(dst,stride,r,nT,bit_depth); }
template <> inline void acceleration_functions::add_residual(uint16_t *dst, ptrdiff_t stride, const int32_t* r, int nT, int bit_depth) const { add_residual_16(dst,stride,r,nT,bit_depth); }

template <> inline void acceleration_functions::add_residual16(uint8_t *dst,  ptrdiff_t stride, const int16_t* r, int nT, int bit_depth) const { add_residual16_8(dst,stride,r,nT,bit_depth); }
template <> inline void acceleration_functions::add_residual16(uint16_t *dst, ptrdiff_t stride, const int16_t* r, int nT, int bit_depth) const { add_residual16_16(dst,stride,r,nT,bit_depth); }

template <> inline void acceleration_functions::intra_pred_dc<uint8_t>(uint8_t* dst, int dstStride, int nT, int log2TrafoSize, int cIdx, uint8_t* border) const { intra_pred_dc_8[log2TrafoSize-2](dst, dstStride, nT,cIdx, border); } 
template <> inline void acceleration_functions::intra_pred_dc<uint16_t>(uint16_t* dst, int dstStride, int nT, int log2TrafoSize, int cIdx, uint16_t* border) const { intra_pred_dc_16[log2TrafoSize-2](dst, dstStride, nT,cIdx, border); }

template <> inline void acceleration_functions::intra_prediction_angular<uint8_t>(int AngIdx, uint8_t *dst, int dstStride, int bit_depth, bool disableIntraBoundaryFilter, int xB0,int yB0, enum IntraPredMode intraPredMode, int nT,int cIdx, uint8_t * border) const { 
                                                intra_prediction_angular_8[AngIdx](dst, dstStride, bit_depth, disableIntraBoundaryFilter, xB0,yB0, intraPredMode, nT,cIdx, border); }
template <> inline void acceleration_functions::intra_prediction_angular<uint16_t>(int AngIdx, uint16_t *dst, int dstStride, int bit_depth, bool disableIntraBoundaryFilter, int xB0,int yB0, enum IntraPredMode intraPredMode, int nT,int cIdx, uint16_t * border) const { 
                                                intra_prediction_angular_16[AngIdx](dst, dstStride, bit_depth, disableIntraBoundaryFilter, xB0,yB0, intraPredMode, nT,cIdx, border); }

template <> inline void acceleration_functions::intra_prediction_sample_filtering<uint8_t>(const seq_parameter_set& sps, uint8_t * p, int nT, int cIdx, enum IntraPredMode intraPredMode) const {intra_prediction_sample_filtering_8(sps, p, nT, cIdx, intraPredMode); }
template <> inline void acceleration_functions::intra_prediction_sample_filtering<uint16_t>(const seq_parameter_set& sps, uint16_t * p, int nT, int cIdx, enum IntraPredMode intraPredMode) const {intra_prediction_sample_filtering_16(sps, p, nT, cIdx, intraPredMode); }

template <> inline void acceleration_functions::intra_prediction_planar<uint8_t>(uint8_t *dst, int dstStride, int nT, int log2TrafoSize, int cIdx, uint8_t *border) const {intra_prediction_planar_8(dst, dstStride, nT, cIdx, border);}
template <> inline void acceleration_functions::intra_prediction_planar<uint16_t>(uint16_t *dst, int dstStride, int nT, int log2TrafoSize, int cIdx, uint16_t *border) const {intra_prediction_planar_16(dst, dstStride, nT, cIdx, border);}

template <> inline void acceleration_functions::loop_filter_luma(uint8_t *dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const {loop_filter_luma_8(dst, vertical, stride, beta, tc, no_p, no_q, bitDepth);}
template <> inline void acceleration_functions::loop_filter_luma(uint16_t *dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const {loop_filter_luma_16(dst, vertical, stride, beta, tc, no_p, no_q, bitDepth);}
template <> inline void acceleration_functions::loop_filter_luma_c(uint8_t *dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const {loop_filter_luma_c_8(dst, vertical, stride, beta, tc, no_p, no_q, bitDepth);}
template <> inline void acceleration_functions::loop_filter_luma_c(uint16_t *dst, bool vertical, ptrdiff_t stride, int beta, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const {loop_filter_luma_c_16(dst, vertical, stride, beta, tc, no_p, no_q, bitDepth);}

#ifdef OPT_4
template <> inline void acceleration_functions::loop_filter_chroma(uint8_t *dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const {loop_filter_chroma_8(dst, vertical, stride, tc, no_p, no_q, bitDepth);}
template <> inline void acceleration_functions::loop_filter_chroma(uint16_t *dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const {loop_filter_chroma_16(dst, vertical, stride, tc, no_p, no_q, bitDepth);}
template <> inline void acceleration_functions::loop_filter_chroma_c(uint8_t *dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const {loop_filter_chroma_c_8(dst, vertical, stride, tc, no_p, no_q, bitDepth);}
template <> inline void acceleration_functions::loop_filter_chroma_c(uint16_t *dst, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth) const {loop_filter_chroma_c_16(dst, vertical, stride, tc, no_p, no_q, bitDepth);}
#else
template <> inline void acceleration_functions::loop_filter_chroma(uint8_t *dst, bool vertical, ptrdiff_t stride, int tc, bool filterP, bool filterQ, int bitDepth) const {loop_filter_chroma_8(dst, vertical, stride, tc, filterP, filterQ, bitDepth);}
template <> inline void acceleration_functions::loop_filter_chroma(uint16_t *dst, bool vertical, ptrdiff_t stride, int tc, bool filterP, bool filterQ, int bitDepth) const {loop_filter_chroma_16(dst, vertical, stride, tc, filterP, filterQ, bitDepth);}
template <> inline void acceleration_functions::loop_filter_chroma_c(uint8_t *dst, bool vertical, ptrdiff_t stride, int tc, bool filterP, bool filterQ, int bitDepth) const {loop_filter_chroma_c_8(dst, vertical, stride, tc, filterP, filterQ, bitDepth);}
template <> inline void acceleration_functions::loop_filter_chroma_c(uint16_t *dst, bool vertical, ptrdiff_t stride, int tc, bool filterP, bool filterQ, int bitDepth) const {loop_filter_chroma_c_16(dst, vertical, stride, tc, filterP, filterQ, bitDepth);}
#endif

template <> inline void acceleration_functions::sao_band_filter(uint8_t *_dst, int stride_dst, /*const*/ uint8_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, int saoLeftClass, const int8_t *saoOffsetVal, int bandshift, const int maxPixelValue) const 
{
   sao_band_filter_8(_dst, stride_dst, _src, stride_src, cIdx, ctuW, ctuH, saoLeftClass, saoOffsetVal, bandshift, maxPixelValue);
}
template <> inline void acceleration_functions::sao_band_filter(uint16_t *_dst, int stride_dst, /*const*/ uint16_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, int saoLeftClass, const int8_t *saoOffsetVal, int bandshift, const int maxPixelValue) const 
{
   sao_band_filter_16(_dst, stride_dst, _src, stride_src, cIdx, ctuW, ctuH, saoLeftClass, saoOffsetVal, bandshift, maxPixelValue);
}

template <> inline void acceleration_functions::sao_edge_filter(uint8_t* _dst, int _stride_dst, uint8_t* _src , int _stride_src, int SaoEoClass, int8_t* saoOffsetVal, int* edges, int ctuW, int ctuH, const int maxPixelValue) const
{
  sao_edge_filter_8(_dst, _stride_dst,  _src , _stride_src, SaoEoClass, saoOffsetVal, edges, ctuW, ctuH, maxPixelValue);
}
template <> inline void acceleration_functions::sao_edge_filter(uint16_t* _dst, int _stride_dst, uint16_t* _src , int _stride_src, int SaoEoClass, int8_t* saoOffsetVal, int* edges, int ctuW, int ctuH, const int maxPixelValue) const
{
  sao_edge_filter_16(_dst, _stride_dst,  _src , _stride_src, SaoEoClass, saoOffsetVal, edges, ctuW, ctuH, maxPixelValue);
}
#endif
