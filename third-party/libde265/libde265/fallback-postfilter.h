#ifndef FALLBACK_PF_H
#define FALLBACK_PF_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "util.h"
#include <iostream>

#define P3 pix[-4 * xstride]
#define P2 pix[-3 * xstride]
#define P1 pix[-2 * xstride]
#define P0 pix[-1 * xstride]
#define Q0 pix[0 * xstride]
#define Q1 pix[1 * xstride]
#define Q2 pix[2 * xstride]
#define Q3 pix[3 * xstride]

// line three. used only for deblocking decision
#define TP3 pix[-4 * xstride + 3 * ystride]
#define TP2 pix[-3 * xstride + 3 * ystride]
#define TP1 pix[-2 * xstride + 3 * ystride]
#define TP0 pix[-1 * xstride + 3 * ystride]
#define TQ0 pix[0  * xstride + 3 * ystride]
#define TQ1 pix[1  * xstride + 3 * ystride]
#define TQ2 pix[2  * xstride + 3 * ystride]
#define TQ3 pix[3  * xstride + 3 * ystride]


template <class pixel_t>
void loop_filter_luma(pixel_t *dst, bool vertical, ptrdiff_t stride, int beta, int*_tc, uint8_t *_no_p, uint8_t *_no_q, int bitDepth){
  ptrdiff_t xstride = vertical ? 1 : stride;
  ptrdiff_t ystride = vertical ? stride : 1;

  // pixel_t* src = vertical ? dst - 4 :  dst - 4*stride;
  // for (size_t j = 0; j < 8; j++)
  // {
  //   for (size_t i = 0; i < 8; i++)
  //   {
  //     std::cout << (int) src[j*stride+i] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;


  pixel_t *pix = dst;

  //beta <<= bitDepth - 8;
    
  for (int j = 0; j < 2; j++) {
    const int dp0  = abs(P2  - 2 * P1  + P0);
    const int dq0  = abs(Q2  - 2 * Q1  + Q0);
    const int dp3  = abs(TP2 - 2 * TP1 + TP0);
    const int dq3  = abs(TQ2 - 2 * TQ1 + TQ0);
    const int d0   = dp0 + dq0;
    const int d3   = dp3 + dq3;
    const int tc   = _tc[j]; //   << (bitDepth - 8);
    const int no_p = _no_p[j];
    const int no_q = _no_q[j];

    if (d0 + d3 >= beta) {
      pix += 4 * ystride;
      continue;
    } else {
      const int beta_3 = beta >> 3;
      const int beta_2 = beta >> 2;
      const int tc25   = ((tc * 5 + 1) >> 1);

      if (abs(P3  -  P0) + abs(Q3  -  Q0) < beta_3 && abs(P0  -  Q0) < tc25 &&
          abs(TP3 - TP0) + abs(TQ3 - TQ0) < beta_3 && abs(TP0 - TQ0) < tc25 &&
                                (d0 << 1) < beta_2 &&      (d3 << 1) < beta_2) {
        // strong filtering
        const int tc2 = tc << 1;
        for (int d = 0; d < 4; d++) {
          const int p3 = P3;
          const int p2 = P2;
          const int p1 = P1;
          const int p0 = P0;
          const int q0 = Q0;
          const int q1 = Q1;
          const int q2 = Q2;
          const int q3 = Q3;
          if (!no_p) {
            P0 = p0 + Clip3(-tc2, tc2, ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) - p0);
            P1 = p1 + Clip3(-tc2, tc2, ((p2 + p1 + p0 + q0 + 2) >> 2) - p1);
            P2 = p2 + Clip3(-tc2, tc2, ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) - p2);
          }
          if (!no_q) {
            Q0 = q0 + Clip3(-tc2, tc2, ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) - q0);
            Q1 = q1 + Clip3(-tc2, tc2, ((p0 + q0 + q1 + q2 + 2) >> 2) - q1);
            Q2 = q2 + Clip3(-tc2, tc2, ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) - q2);
          }
          pix += ystride;
        }
      } else { // normal filtering
        int nd_p = 1;
        int nd_q = 1;
        const int tc_2 = tc >> 1;
        if (dp0 + dp3 < ((beta + (beta >> 1)) >> 3))
            nd_p = 2;
        if (dq0 + dq3 < ((beta + (beta >> 1)) >> 3))
            nd_q = 2;

        for (int d = 0; d < 4; d++) {
            const int p2 = P2;
            const int p1 = P1;
            const int p0 = P0;
            const int q0 = Q0;
            const int q1 = Q1;
            const int q2 = Q2;
            int delta0   = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if (abs(delta0) < 10 * tc) {
                delta0 = Clip3(-tc, tc, delta0);
                if (!no_p)
                    P0 = Clip_BitDepth(p0 + delta0, bitDepth);
                if (!no_q)
                    Q0 = Clip_BitDepth(q0 - delta0, bitDepth);
                if (!no_p && nd_p > 1) {
                    const int deltap1 = Clip3(-tc_2, tc_2, (((p2 + p0 + 1) >> 1) - p1 + delta0) >> 1);
                    P1 = Clip_BitDepth(p1 + deltap1, bitDepth);
                }
                if (!no_q && nd_q > 1) {
                    const int deltaq1 = Clip3(-tc_2, tc_2, (((q2 + q0 + 1) >> 1) - q1 - delta0) >> 1);
                    Q1 = Clip_BitDepth(q1 + deltaq1, bitDepth);
                }
            }
            pix += ystride;
        }
      }
    }
  }
}

#ifdef OPT_4
template <class pixel_t>
void loop_filter_chroma(pixel_t *ptr, bool vertical, ptrdiff_t stride, int* tc, uint8_t *no_p, uint8_t *no_q, int bitDepth_C){
  pixel_t p[2][8];
  pixel_t q[2][8];

  for (int i=0;i<2;i++){
    for (int k=0;k<8;k++)
    {
      if (vertical) {
        q[i][k] = ptr[ i  +k*stride];
        p[i][k] = ptr[-i-1+k*stride];
      }
      else {
        q[i][k] = ptr[k + i   *stride];
        p[i][k] = ptr[k -(i+1)*stride];
      }
    }
  }
  if (vertical) {
    for (int k=0;k<4;k++) {
      int delta = Clip3(-tc[0],tc[0], ((((q[0][k]-p[0][k])*4)+p[1][k]-q[1][k]+4)>>3)); // standard says <<2 in eq. (8-356), but the value can also be negative
      if (no_p[0]) { ptr[-1+k*stride] = Clip_BitDepth(p[0][k]+delta, bitDepth_C); }
      if (no_p[0]) { ptr[ 0+k*stride] = Clip_BitDepth(q[0][k]-delta, bitDepth_C); }
    }
    for (int k=4;k<8;k++) {
      int delta = Clip3(-tc[1],tc[1], ((((q[0][k]-p[0][k])*4)+p[1][k]-q[1][k]+4)>>3)); // standard says <<2 in eq. (8-356), but the value can also be negative
      if (no_p[1]) { ptr[-1+k*stride] = Clip_BitDepth(p[0][k]+delta, bitDepth_C); }
      if (no_p[1]) { ptr[ 0+k*stride] = Clip_BitDepth(q[0][k]-delta, bitDepth_C); }
    }
  }
  else{
    for (int k=0;k<4;k++) {
      int delta = Clip3(-tc[0],tc[0], ((((q[0][k]-p[0][k])*4)+p[1][k]-q[1][k]+4)>>3)); // standard says <<2, but the value can also be negative
      if (no_p[0]) { ptr[ k-1*stride] = Clip_BitDepth(p[0][k]+delta, bitDepth_C); }
      if (no_q[0]) { ptr[ k+0*stride] = Clip_BitDepth(q[0][k]-delta, bitDepth_C); }
    }
    for (int k=4;k<8;k++) {
      int delta = Clip3(-tc[1],tc[1], ((((q[0][k]-p[0][k])*4)+p[1][k]-q[1][k]+4)>>3)); // standard says <<2, but the value can also be negative
      if (no_p[1]) { ptr[ k-1*stride] = Clip_BitDepth(p[0][k]+delta, bitDepth_C); }
      if (no_q[1]) { ptr[ k+0*stride] = Clip_BitDepth(q[0][k]-delta, bitDepth_C); }
    }
  }
}
#else
template <class pixel_t>
void loop_filter_chroma(pixel_t *ptr, bool vertical, ptrdiff_t stride, int tc, bool filterP, bool filterQ, int bitDepth_C){
  pixel_t p[2][4];
  pixel_t q[2][4];

  for (int i=0;i<2;i++){
    for (int k=0;k<4;k++)
    {
      if (vertical) {
        q[i][k] = ptr[ i  +k*stride];
        p[i][k] = ptr[-i-1+k*stride];
      }
      else {
        q[i][k] = ptr[k + i   *stride];
        p[i][k] = ptr[k -(i+1)*stride];
      }
    }
  }
  if (vertical) {
    for (int k=0;k<4;k++) {
      int delta = Clip3(-tc,tc, ((((q[0][k]-p[0][k])*4)+p[1][k]-q[1][k]+4)>>3)); // standard says <<2 in eq. (8-356), but the value can also be negative
      if (filterP) { ptr[-1+k*stride] = Clip_BitDepth(p[0][k]+delta, bitDepth_C); }
      if (filterQ) { ptr[ 0+k*stride] = Clip_BitDepth(q[0][k]-delta, bitDepth_C); }
    }
  }
  else{
    for (int k=0;k<4;k++) {
      int delta = Clip3(-tc,tc, ((((q[0][k]-p[0][k])*4)+p[1][k]-q[1][k]+4)>>3)); // standard says <<2, but the value can also be negative
      if (filterP) { ptr[ k-1*stride] = Clip_BitDepth(p[0][k]+delta, bitDepth_C); }
      if (filterQ) { ptr[ k+0*stride] = Clip_BitDepth(q[0][k]-delta, bitDepth_C); }
    }
  }
}
#endif

template <class pixel_t>
void sao_band_filter_c(pixel_t *_dst, int stride_dst, /*const*/ pixel_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, int saoLeftClass, const int8_t *saoOffsetVal, int bandShift, const int maxPixelValue){
    
  int bandTable[32];
  memset(bandTable, 0, sizeof(int)*32);

  for (int k=0;k<4;k++) {
    bandTable[ (k+saoLeftClass)&31 ] = k+1;
  }

  for (int j=0;j<ctuH;j++){
    for (int i=0;i<ctuW;i++) {
      int bandIdx;
      if (bandShift >= 8) {
        bandIdx = 0;
      } else {
        bandIdx = bandTable[ _src[i+j*stride_src]>>bandShift ];
      }
      if (bandIdx>0) {
        int offset = saoOffsetVal[bandIdx-1];
        _dst[i + j * stride_dst] = Clip3(0, maxPixelValue, _src[i + j*stride_src] + offset);
      }
    }
  }
}

template <class pixel_t>
void sao_edge_filter_c(pixel_t* out_ptr, int out_stride, pixel_t* in_ptr , int in_stride, int SaoEoClass, int8_t* saoOffsetVal, int* edges, int ctuW, int ctuH, const int maxPixelValue){
  int hPos[2], vPos[2];
  int vPosStride[2]; // vPos[] multiplied by image stride
  switch (SaoEoClass) {
    case 0: hPos[0]=-1; hPos[1]= 1; vPos[0]= 0; vPos[1]=0; break;
    case 1: hPos[0]= 0; hPos[1]= 0; vPos[0]=-1; vPos[1]=1; break;
    case 2: hPos[0]=-1; hPos[1]= 1; vPos[0]=-1; vPos[1]=1; break;
    case 3: hPos[0]= 1; hPos[1]=-1; vPos[0]=-1; vPos[1]=1; break;
  }
  vPosStride[0] = vPos[0] * in_stride;
  vPosStride[1] = vPos[1] * in_stride;
  
  for (int j=0;j<ctuH;j++) {
    pixel_t* src = in_ptr + j * in_stride;
    pixel_t* dst = out_ptr +j * out_stride;
    for (int i=0;i<ctuW;i++) {
      int edgeIdx = -1;
      logtrace(LogSAO, "pos %d,%d\n",xC+i,yC+j);
      edgeIdx = ( Sign(src[i] - src[i+hPos[0]+vPosStride[0]]) +
                  Sign(src[i] - src[i+hPos[1]+vPosStride[1]])   );

      int offset = saoOffsetVal[edgeIdx+2];
      dst[i] = Clip3(0,maxPixelValue, src[i] + offset);
    }
  }

  int init_x = 0;
  int last_x = ctuW;
  if (SaoEoClass!=1)
  {
    if (edges[0])
    {
      int offset_val = saoOffsetVal[2];
      int y_stride_src   = 0;
      int y_stride_dst   = 0;
      for (int y = 0; y < ctuH; y++) {
        out_ptr[y_stride_dst] =  Clip3(0,maxPixelValue, in_ptr[y_stride_src] + offset_val);
        y_stride_src     += in_stride;
        y_stride_dst     += out_stride;
      }
      init_x = 1;
    }
    if (edges[2])
    {
      int offset_val = saoOffsetVal[2];
      int y_stride_src   = last_x - 1;
      int y_stride_dst   = last_x - 1;
      for (int y = 0; y < ctuH; y++) {
        out_ptr[y_stride_dst] = Clip3(0,maxPixelValue, in_ptr[y_stride_src] + offset_val);
        y_stride_src     += in_stride;
        y_stride_dst     += out_stride;
      }
      last_x--;
    }
  }
  if (SaoEoClass!=0)
  {
    if (edges[1])
    {
      int offset_val = saoOffsetVal[2];
      for (int x = init_x; x < last_x; x++)
        out_ptr[x] = Clip3(0,maxPixelValue, in_ptr[x] + offset_val);
    }
    if (edges[3])
    {
      int offset_val = saoOffsetVal[2];
      int y_stride_src   = in_stride * (ctuH - 1);
      int y_stride_dst   = out_stride * (ctuH - 1);
      for (int x = init_x; x < last_x; x++)
        out_ptr[x + y_stride_dst] = Clip3(0,maxPixelValue, in_ptr[x + y_stride_src] + offset_val);
    }
  }
}
#endif