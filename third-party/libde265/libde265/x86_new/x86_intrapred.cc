#include "x86_intrapred.h"
#include "libde265/util.h"
#include "../intrapred.h"

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
void intra_prediction_DC_4_8_sse4(uint8_t *_dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border)
{
    //INIT
    uint8_t *dst = _dst;
    uint8_t* top = border+1;
    uint8_t* left = border-nT;

    //Load Border
    __m128i val0 = _mm_loadl_epi64((__m128i *) (top)); 
            val0 = _mm_unpacklo_epi8(val0, _mm_setzero_si128());
    __m128i val1 = _mm_loadl_epi64((__m128i *) (left)); 
            val1 = _mm_unpacklo_epi8(val1, _mm_setzero_si128());
    __m128i val  = _mm_unpacklo_epi16(val0, val1);

    //sum dc
    //val = _mm_set_epi16(8,7,6,5,4,3,2,1);
    val = _mm_hadd_epi16(val, val);
    val = _mm_hadd_epi16(val, val);
    val = _mm_unpacklo_epi16(val, _mm_setzero_si128());
    val = _mm_hadd_epi32(val, val);
    int32_t DcVal = (_mm_cvtsi128_si32(val) + nT) >> 3;


    //cpy dc to all pixels
    for (int i = 0; i < nT; i++)
    {
        memset(dst, DcVal, 4*sizeof(uint8_t));
        dst +=dstStride ;
    }

    if(cIdx==0)
    {
        dst = _dst;
        int16_t dcval_dup= 3*DcVal+2 ;

        //cpy to first line
        __m128i dat = _mm_set1_epi16(dcval_dup);
                val = _mm_loadl_epi64((__m128i *) (top)); 
                val = _mm_unpacklo_epi8(val, _mm_setzero_si128());
                val = _mm_add_epi16(dat, val);
                val = _mm_srai_epi16(val, 2);
                val = _mm_and_si128(val, _mm_set1_epi16(0xff)); //to avoid saturation
                val = _mm_packus_epi16 (val,val); //narrow, use low 64 bits only
        int32_t res = _mm_cvtsi128_si32(val); 
        memcpy(dst, &res, sizeof(res));

    
        //cp to first colum
        for (int y=1;y<nT;y++) 
        { 
                dst[y*dstStride] = (border[-y-1] + dcval_dup)>>2; 
        }

        //cpy to zero position
        dst[0] = (border[-1] + 2*DcVal + border[1] + 2) >> 2;
     }
}

void intra_prediction_DC_8_8_sse4(uint8_t *_dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border)
{
    //INIT
    uint8_t *dst = _dst;
    uint8_t* top = border+1;
    uint8_t* left = border-nT;

    //Load Border
    __m128i val[2];
            val[0] = _mm_loadl_epi64((__m128i *) (top)); 
            val[0] = _mm_unpacklo_epi8(val[0], _mm_setzero_si128());
            val[1] = _mm_loadl_epi64((__m128i *) (left)); 
            val[1] = _mm_unpacklo_epi8(val[1], _mm_setzero_si128());

    //sum dc
    __m128i tmp[2];
    for (int i = 0; i < 2; i++)
    {
        tmp[i] = _mm_hadd_epi16(val[i], val[i]);
        tmp[i] = _mm_hadd_epi16(tmp[i], tmp[i]);
        tmp[i] = _mm_unpacklo_epi16(tmp[i], _mm_setzero_si128());
        tmp[i] = _mm_hadd_epi32(tmp[i], tmp[i]);
    }
    __m128i value = _mm_add_epi32(tmp[0], tmp[1]);

    //average
    int32_t DcVal = (_mm_cvtsi128_si32(value) + nT) >> 4;

    //cpy dc to all pixels
    value = _mm_set1_epi8(DcVal);
    for (int i = 0; i < nT; i++)
    {
        _mm_storel_epi64((__m128i *) dst, value);
        dst +=dstStride ;
    }

    if(cIdx==0)
    {
        dst = _dst;
        int16_t dcval_dup= 3*DcVal+2 ;

        //cpy to first line
        __m128i dat = _mm_set1_epi16(dcval_dup);
        __m128i val = _mm_loadl_epi64((__m128i *) (top)); 
                val = _mm_unpacklo_epi8(val, _mm_setzero_si128());
                val = _mm_add_epi16(dat, val);
                val = _mm_srai_epi16(val, 2);
                val = _mm_and_si128(val, _mm_set1_epi16(0xff)); //to avoid saturation
                val = _mm_packus_epi16 (val,val); //narrow, use low 64 bits only
        _mm_storel_epi64((__m128i *) dst, val);

        //cp to first colum
        for (int y=1;y<nT;y++) 
        { 
                dst[y*dstStride] = (border[-y-1] + dcval_dup)>>2; 
        }

        //cpy to zero position
        dst[0] = (border[-1] + 2*DcVal + border[1] + 2) >> 2;
      }
}

void intra_prediction_DC_16_8_sse4(uint8_t *_dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border)
{
    //INIT
    uint8_t *dst = _dst;
    uint8_t* top = border+1;
    uint8_t* left = border-nT;

    //Load Border
    __m128i val[4];
    __m128i val_t  = _mm_loadu_si128((__m128i *) (top)); 
            val[0] = _mm_unpacklo_epi8(val_t, _mm_setzero_si128());
            val[1] = _mm_unpackhi_epi8(val_t, _mm_setzero_si128());
    __m128i val_l  = _mm_loadu_si128((__m128i *) (left)); 
            val[2] = _mm_unpacklo_epi8(val_l, _mm_setzero_si128());
            val[3] = _mm_unpackhi_epi8(val_l, _mm_setzero_si128());

    //sum dc
    __m128i tmp[4];
    for (int i = 0; i < 4; i++)
    {
        tmp[i] = _mm_hadd_epi16(val[i], val[i]);
        tmp[i] = _mm_hadd_epi16(tmp[i], tmp[i]);
        tmp[i] = _mm_unpacklo_epi16(tmp[i], _mm_setzero_si128());
        tmp[i] = _mm_hadd_epi32(tmp[i], tmp[i]);
    }
    __m128i value0 = _mm_add_epi32(tmp[0], tmp[1]);
    __m128i value1 = _mm_add_epi32(tmp[2], tmp[3]);
    __m128i value  = _mm_add_epi32(value0, value1);

    //average
    int32_t DcVal = (_mm_cvtsi128_si32(value) + nT) >> 5;

    //cpy dc to all pixels
    value = _mm_set1_epi8(DcVal);
    for (int i = 0; i < nT; i++)
    {
        _mm_storeu_si128((__m128i *) dst, value);
        dst +=dstStride ;
    }

    if(cIdx==0)
    {
        dst = _dst;
        int16_t dcval_dup= 3*DcVal+2 ;

        //cpy to first line
        for (int idx = 0; idx < nT; idx+=8)
        {
                __m128i dat = _mm_set1_epi16(dcval_dup);
                __m128i val = _mm_loadl_epi64((__m128i *) (top+idx)); 
                        val = _mm_unpacklo_epi8(val, _mm_setzero_si128());
                        val = _mm_add_epi16(dat, val);
                        val = _mm_srai_epi16(val, 2);
                        val = _mm_and_si128(val, _mm_set1_epi16(0xff)); //to avoid saturation
                        val = _mm_packus_epi16 (val,val); //narrow, use low 64 bits only
                _mm_storel_epi64((__m128i *) (dst+idx), val);
        }
        
        //cp to first colum
        for (int y=1;y<nT;y++) 
        { 
                dst[y*dstStride] = (border[-y-1] + dcval_dup)>>2; 
        }

        //cpy to zero position
        dst[0] = (border[-1] + 2*DcVal + border[1] + 2) >> 2;

      }
}


void intra_prediction_DC_32_8_sse4(uint8_t *_dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border)
{
    //INIT
    uint8_t *dst = _dst;
    uint8_t* top = border+1;
    uint8_t* left = border-nT;

    //Load Border
    __m128i val[8];
    __m128i val_t  = _mm_loadu_si128((__m128i *) (top)); 
            val[0] = _mm_unpacklo_epi8(val_t, _mm_setzero_si128());
            val[1] = _mm_unpackhi_epi8(val_t, _mm_setzero_si128());
            val_t  = _mm_loadu_si128((__m128i *) (top+16)); 
            val[2] = _mm_unpacklo_epi8(val_t, _mm_setzero_si128());
            val[3] = _mm_unpackhi_epi8(val_t, _mm_setzero_si128());

    __m128i val_l  = _mm_loadu_si128((__m128i *) (left)); 
            val[4] = _mm_unpacklo_epi8(val_l, _mm_setzero_si128());
            val[5] = _mm_unpackhi_epi8(val_l, _mm_setzero_si128());
            val_l  = _mm_loadu_si128((__m128i *) (left+16)); 
            val[6] = _mm_unpacklo_epi8(val_l, _mm_setzero_si128());
            val[7] = _mm_unpackhi_epi8(val_l, _mm_setzero_si128());

    //sum dc
    __m128i tmp[8];
    for (int i = 0; i < 8; i++)
    {
        tmp[i] = _mm_hadd_epi16(val[i], val[i]);
        tmp[i] = _mm_hadd_epi16(tmp[i], tmp[i]);
        tmp[i] = _mm_unpacklo_epi16(tmp[i], _mm_setzero_si128());
        tmp[i] = _mm_hadd_epi32(tmp[i], tmp[i]);
    }
    __m128i value0 = _mm_add_epi32(tmp[0], tmp[1]);
    __m128i value1 = _mm_add_epi32(tmp[2], tmp[3]);
    __m128i value2 = _mm_add_epi32(tmp[4], tmp[5]);
    __m128i value3 = _mm_add_epi32(tmp[6], tmp[7]);
    __m128i value  = _mm_add_epi32(value0, value1);
            value  = _mm_add_epi32(value, value2);
            value  = _mm_add_epi32(value, value3);


    //average
    int32_t DcVal = (_mm_cvtsi128_si32(value) + nT) >> 6;

    //cpy dc to all pixels
    value = _mm_set1_epi8(DcVal);
    for (int i = 0; i < nT; i++)
    {
        _mm_storeu_si128((__m128i *) dst, value);
        _mm_storeu_si128((__m128i *) (dst+16), value);
        dst +=dstStride ;
    }

}


extern const int intraPredAngle_table[1+34];


LIBDE265_INLINE __m128i vrshrn16(__m128i a, int b) // VRSHRN.I16 d0,q0,#8
{
    __m128i mask, r16;
    mask = _mm_set1_epi16(0xff);
    __m128i maskb =  _mm_slli_epi16(a, (16 - b));
    maskb = _mm_srli_epi16(maskb, 15); 
    r16  = _mm_srai_epi16(a,b); //after right shift b>=1 unsigned var fits into signed range, so we could use _mm_packus_epi16 (signed 16 to unsigned 8)
    r16 = _mm_add_epi16 (r16, maskb);
    r16 = _mm_and_si128(r16, mask); //to avoid saturation
    r16 = _mm_packus_epi16 (r16,r16); //saturate and  narrow, use low 64 bits only
    return (r16);
}

// angle 27 ~ 34
void intra_prediction_angular_27_34_sse4(uint8_t* dst, int dstStride,
                                         int bit_depth, bool disableIntraBoundaryFilter,
                                         int xB0,int yB0,
                                         enum IntraPredMode intraPredMode,
                                         int nT,int cIdx,
                                         uint8_t * border)
{

  int intraPredAngle = intraPredAngle_table[intraPredMode];

  // 4x4
  if(nT == 4) {

    int y = 0;
    do {
      int iIdx = ((y+1)*intraPredAngle)>>5 ;
      int iFact= ((y+1)*intraPredAngle)&31 ;

      __m128i vref_l = _mm_loadl_epi64((__m128i *) (border+iIdx+1));  //8x16  least 8x8
              vref_l = _mm_unpacklo_epi8(vref_l, _mm_setzero_si128());  //16x8
      __m128i vref_r = _mm_loadl_epi64((__m128i *) (border+iIdx+2));  //8x16  least 8x8
              vref_r = _mm_unpacklo_epi8(vref_r, _mm_setzero_si128());  //16x8

      __m128i weight_a = _mm_mullo_epi16(vref_l, _mm_set1_epi16(32-(uint8_t)iFact));
      __m128i weight_s = _mm_mullo_epi16(vref_r, _mm_set1_epi16(iFact));
              weight_s = _mm_add_epi16(weight_a, weight_s);

      __m128i val = vrshrn16(weight_s, 5);

      *((uint32_t *)(dst+y*dstStride)) = _mm_cvtsi128_si32(val);
    } while (++y < nT);
  }
  else {

    int y = 0;
    do {
      int iIdx = ((y+1)*intraPredAngle)>>5 ;
      int iFact= ((y+1)*intraPredAngle)&31 ;
      int lidx = 0 ;

      do {
              __m128i vref_l = _mm_loadl_epi64((__m128i *) (border+lidx*8+iIdx+1));  //8x16  least 8x8
                      vref_l = _mm_unpacklo_epi8(vref_l, _mm_setzero_si128());  //16x8

              __m128i vref_r = _mm_loadl_epi64((__m128i *) (border+lidx*8+iIdx+2));  //8x16  least 8x8
                      vref_r = _mm_unpacklo_epi8(vref_r, _mm_setzero_si128());  //16x8

             __m128i weight_a = _mm_mullo_epi16(vref_l, _mm_set1_epi16(32-(uint8_t)iFact));

             __m128i weight_s = _mm_mullo_epi16(vref_r, _mm_set1_epi16(iFact));
                     weight_s = _mm_add_epi16(weight_a, weight_s);

             __m128i val = vrshrn16(weight_s, 5);

             _mm_storel_epi64((__m128i *) (dst+y*dstStride+lidx*8), val);

      } while (++lidx < (nT/8)) ;

    } while (++y < nT) ;
  }

  return ;
}


// angle 18 ~ 26
void intra_prediction_angular_18_26_sse4(uint8_t* dst, int dstStride,
                                         int bit_depth, bool disableIntraBoundaryFilter,
                                         int xB0,int yB0,
                                         enum IntraPredMode intraPredMode,
                                         int nT,int cIdx,
                                         uint8_t * border)
{
  uint8_t ref_mem[4*MAX_INTRA_PRED_BLOCK_SIZE+1]; // TODO: what is the required range here ?
  uint8_t *ref=&ref_mem[2*MAX_INTRA_PRED_BLOCK_SIZE];

  int intraPredAngle = intraPredAngle_table[intraPredMode];
  int invAngle = invAngle_table[intraPredMode-11];

  // prepare ref pixel
  for (int x=0;x<=nT;x++)
    { ref[x] = border[x]; }

  int s_ref = (nT*intraPredAngle)>>5;
  if (s_ref < -1) {
    for (int x= s_ref; x<=-1; x++) {
      ref[x] = border[0-((x*invAngle+128)>>8)];
    }
  }

  // 4x4
  if(nT == 4) {

    int y = 0;
    do {
      int iIdx = ((y+1)*intraPredAngle)>>5 ;
      int iFact= ((y+1)*intraPredAngle)&31 ;

       __m128i vref_l = _mm_loadl_epi64((__m128i *) (ref+iIdx+1));  //8x16  least 8x8
               vref_l = _mm_unpacklo_epi8(vref_l, _mm_setzero_si128());  //16x8
       __m128i vref_r = _mm_loadl_epi64((__m128i *) (ref+iIdx+2));  //8x16  least 8x8
               vref_r = _mm_unpacklo_epi8(vref_r, _mm_setzero_si128());  //16x8

        __m128i weight_a = _mm_mullo_epi16(vref_l, _mm_set1_epi16(32-(uint8_t)iFact));
        __m128i weight_s = _mm_mullo_epi16(vref_r, _mm_set1_epi16(iFact));
                weight_s = _mm_add_epi16(weight_a, weight_s);

        __m128i val = vrshrn16(weight_s, 5);
        *((uint32_t *)(dst+y*dstStride)) = _mm_cvtsi128_si32(val);

    } while (++y < nT);
  }
  else {

    int y = 0;
    do {
      int iIdx = ((y+1)*intraPredAngle)>>5 ;
      int iFact= ((y+1)*intraPredAngle)&31 ;
      int lidx = 0 ;

      do {
              __m128i vref_l = _mm_loadl_epi64((__m128i *) (ref+lidx*8+iIdx+1));  //8x16  least 8x8
                      vref_l = _mm_unpacklo_epi8(vref_l, _mm_setzero_si128());  //16x8
              __m128i vref_r = _mm_loadl_epi64((__m128i *) (ref+lidx*8+iIdx+2));  //8x16  least 8x8
                      vref_r = _mm_unpacklo_epi8(vref_r, _mm_setzero_si128());  //16x8


             __m128i weight_a = _mm_mullo_epi16(vref_l, _mm_set1_epi16(32-(uint8_t)iFact));
             __m128i weight_s = _mm_mullo_epi16(vref_r, _mm_set1_epi16(iFact));
                     weight_s = _mm_add_epi16(weight_a, weight_s);

             __m128i val = vrshrn16(weight_s, 5);

             _mm_storel_epi64((__m128i *) (dst+y*dstStride+lidx*8), val);

      } while (++lidx < (nT/8)) ;

    } while (++y < nT) ;
  }

  if (intraPredMode==26 && cIdx==0 && nT<32 && !disableIntraBoundaryFilter) {
    for (int y=0;y<nT;y++) {
      dst[0+y*dstStride] = Clip_BitDepth(border[1] + ((border[-1-y] - border[0])>>1), bit_depth);
    }
  }

  return ;
}


LIBDE265_INLINE __m128i vtbl1(__m128i a, __m128i b)
{
    __m128i c7, maskgt, bmask;
    c7 = _mm_set1_epi8 (7);
    maskgt = _mm_cmpgt_epi8(b,c7);
    bmask = _mm_or_si128(b,maskgt);
    bmask = _mm_shuffle_epi8(a,bmask);
    return bmask;
}



LIBDE265_INLINE __m128i vtbl2_u8(__m128i a0, __m128i a1, __m128i b)
{
   // uint8x8_t res64;
    __m128i c15, a01, maskgt15, bmask;
    c15 = _mm_set1_epi8 (15);
    maskgt15 = _mm_cmpgt_epi8(b,c15);
    bmask = _mm_or_si128(b, maskgt15);
    a01 = _mm_unpacklo_epi64(a0, a1);
    a01 =  _mm_shuffle_epi8(a01, bmask);
    return a01;
}


LIBDE265_INLINE __m128i vshrn16(__m128i a, int b) // VRSHRN.I16 d0,q0,#8
{
    __m128i mask, r16;
    mask = _mm_set1_epi16(0xff); 
    r16  = _mm_srai_epi16(a,b); //after right shift b>=1 unsigned var fits into signed range, so we could use _mm_packus_epi16 (signed 16 to unsigned 8)
    r16 = _mm_and_si128(r16, mask); //to avoid saturation
    r16 = _mm_packus_epi16 (r16,r16); //saturate and  narrow, use low 64 bits only
    return (r16);
}

LIBDE265_INLINE __m128i vtbl4(__m128i a0, __m128i a1, __m128i a2, __m128i a3, __m128i b)
{
    __m128i c15, c31, maskgt31, bmask, maskgt15, sh0, sh1, a01, a23, b128;
    c15 = _mm_set1_epi8 (15);
    c31 = _mm_set1_epi8 (31);
    maskgt31 = _mm_cmpgt_epi8(b,c31);
    bmask = _mm_or_si128(b, maskgt31);
    maskgt15 = _mm_cmpgt_epi8(b,c15);
    a01 = _mm_unpacklo_epi64(a0, a1);
    a23 = _mm_unpacklo_epi64(a2, a3);
    sh0 =  _mm_shuffle_epi8(a01, bmask);
    sh1 =  _mm_shuffle_epi8(a23, bmask); //for bi>15 bi is wrapped (bi-=15)
    sh0 = _mm_blendv_epi8 (sh0, sh1, maskgt15); //SSE4.1
    return sh0;
}


// angle 10 ~ 17
void intra_prediction_angular_10_17_sse4(uint8_t* dst, int dstStride,
                                         int bit_depth, bool disableIntraBoundaryFilter,
                                         int xB0,int yB0,
                                         enum IntraPredMode intraPredMode,
                                         int nT,int cIdx,
                                         uint8_t * border)
{
  static const int8_t mask8_16_even_odd[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7,  9, 11, 13, 15 };
  uint8_t  ref_mem[4*MAX_INTRA_PRED_BLOCK_SIZE+1]; // TODO: what is the required range here ?
  uint8_t* ref=&ref_mem[2*MAX_INTRA_PRED_BLOCK_SIZE];

  int intraPredAngle = intraPredAngle_table[intraPredMode];
  int invAngle = invAngle_table[intraPredMode-11];

  // prepare ref pixel
  for (int x=0;x<=nT;x++)
    { ref[x] = border[-x]; }

  int s_ref = (nT*intraPredAngle)>>5;
  if (s_ref < -1) {
    for (int x= s_ref; x<=-1; x++) {
      ref[x] = border[((x*invAngle+128)>>8)]; // DIFF (neg)
    }
  }

  // 4x4
  if(nT == 4) {

    int y = 0;
    const __m128i  vidx_x = _mm_set_epi16(0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01);  //16x8
        __m128i  vidx_lx = _mm_mullo_epi16(vidx_x, _mm_set1_epi16(intraPredAngle)); //16x8
        __m128i    viIdx = vshrn16(vidx_lx,5) ; //8x32  least 8x8
                   viIdx = _mm_add_epi8(viIdx, _mm_set1_epi8(nT));
        __m128i    viFact = _mm_and_si128(vidx_lx, _mm_set1_epi16(31));
                   viFact = _mm_shuffle_epi8 (viFact, *(__m128i*) mask8_16_even_odd); 
                   viFact = _mm_unpacklo_epi8(viFact, _mm_setzero_si128());  //16x8
        __m128i    vsFact = _mm_sub_epi16(_mm_set1_epi16(32), viFact); //16x8

    do {
        __m128i tref_l = _mm_loadl_epi64((__m128i *) (ref+y-nT+1));  //least 8x8 avaliable
        __m128i vref_l = vtbl1(tref_l,viIdx); //8x32 least 8x8 avaliable
                vref_l = _mm_unpacklo_epi8(vref_l, _mm_setzero_si128()); //16bits
        __m128i tref_r = _mm_loadl_epi64((__m128i *) (ref+y-nT+2)); 
        __m128i vref_r = vtbl1(tref_r,viIdx);
                vref_r = _mm_unpacklo_epi8(vref_r, _mm_setzero_si128()); //16bits

        __m128i weight_a = _mm_mullo_epi16 (vref_l,vsFact);
        __m128i tmp = _mm_mullo_epi16(vref_r,viFact);
        __m128i weight_s = _mm_add_epi16 (tmp, weight_a);

        __m128i val = vrshrn16(weight_s ,5);
        *((uint32_t *)(dst+y*dstStride)) = _mm_cvtsi128_si32(val);
    } while (++y < nT);
  }
  else if(nT == 8) {

    int y = 0;
    int8_t     iIdx   = (nT*intraPredAngle)>>5;  // iIdx < nT
    const __m128i  vidx_x = _mm_set_epi16(0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01);  //16x8
          __m128i  vidx_lx = _mm_mullo_epi16(vidx_x, _mm_set1_epi16(intraPredAngle)); //16x8
          __m128i  viIdx = vshrn16(vidx_lx,5) ; //8x32  least 8x8
                   viIdx = _mm_sub_epi8(viIdx, _mm_set1_epi8(iIdx));
          __m128i viFact  = _mm_and_si128(vidx_lx, _mm_set1_epi16(31));  //16x8
          __m128i vsFact  = _mm_sub_epi16(_mm_set1_epi16(32), viFact); //16x8

    do {
        int lidx = 0 ;
        __m128i tref_l = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+1)); //8x16  least 8x8
        __m128i tref_r = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+2)); //8x16  least 8x8
        __m128i vref_l = vtbl1(tref_l,viIdx); //least 8x8 avaliable
                vref_l = _mm_unpacklo_epi8(vref_l, _mm_setzero_si128()); //16x8
        __m128i vref_r = vtbl1(tref_r,viIdx);//least 8x8 avaliable
                vref_r = _mm_unpacklo_epi8(vref_r, _mm_setzero_si128()); //16x8

        __m128i weight_a = _mm_mullo_epi16 (vref_l,vsFact);
        __m128i tmp = _mm_mullo_epi16(vref_r,viFact);
        __m128i weight_s = _mm_add_epi16 (tmp, weight_a);
        __m128i val = vrshrn16(weight_s ,5);

        _mm_storel_epi64((__m128i *) (dst+y*dstStride), val);

    } while (++y < nT) ;
  }
  else if(nT == 16) {

    int y = 0;
    int8_t     iIdx   = (nT*intraPredAngle)>>5;  // iIdx < nT
    __m128i   vidx_x = _mm_set1_epi16(0);
    do {
      int lidx = 0 ;
      __m128i tref_l0 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+1));  //8x16  least 8x8
      __m128i tref_l1 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+1+8));  //8x16  least 8x8
      __m128i tref_r0 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+2));  //8x16  least 8x8
      __m128i tref_r1 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+2+8));  //8x16  least 8x8

      do {
        switch(lidx) {
          case 0 : vidx_x = _mm_set_epi16(0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01); break;
          case 1 : vidx_x = _mm_set_epi16(0x10, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09); break;
        }
        __m128i vidx_lx = _mm_mullo_epi16(vidx_x, _mm_set1_epi16(intraPredAngle));
        __m128i viIdx   = vshrn16(vidx_lx, 5); //8x16 least 8x8
                viIdx   = _mm_sub_epi8(viIdx, _mm_set1_epi8(iIdx)); //8x16 least 8x8
               
        __m128i viFact  = _mm_and_si128(vidx_lx, _mm_set1_epi16(31));  //16x8
                viFact  = _mm_shuffle_epi8 (viFact, *(__m128i*) mask8_16_even_odd);  //8x8
                viFact  = _mm_unpacklo_epi8(viFact, _mm_setzero_si128()); //16x8
        __m128i vsFact  = _mm_sub_epi16(_mm_set1_epi16(32), viFact); //16x8

        __m128i vref_l  = vtbl2_u8(tref_l0, tref_l1, viIdx);  //8x8
                vref_l  = _mm_unpacklo_epi8(vref_l, _mm_setzero_si128()); //16x8
        __m128i vref_r  = vtbl2_u8(tref_r0, tref_r1, viIdx);
                vref_r  = _mm_unpacklo_epi8(vref_r, _mm_setzero_si128()); //16x8

        __m128i weight_a =_mm_mullo_epi16 (vref_l,vsFact);
        __m128i weight_s = _mm_mullo_epi16 (vref_r,viFact);
                weight_s = _mm_add_epi16(weight_a, weight_s);

        __m128i val = vrshrn16(weight_s, 5);
        _mm_storel_epi64((__m128i *) (dst+y*dstStride+lidx*8), val);

      } while (++lidx < (nT/8)) ;

    } while (++y < nT) ;
  }
  else {
    assert(nT == 32);
    int y = 0;
    int8_t     iIdx   = (nT*intraPredAngle)>>5;  // iIdx < nT
    __m128i   vidx_x = _mm_set1_epi16(0);
    do {
      int lidx = 0 ;

      __m128i tref_l0 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+1));  //8x16  least 8x8
      __m128i tref_l1 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+1+8));  //8x16  least 8x8
      __m128i tref_l2 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+1+16));  //8x16  least 8x8
      __m128i tref_l3 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+1+24));  //8x16  least 8x8

      __m128i tref_r0 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+2));  //8x16  least 8x8
      __m128i tref_r1 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+2+8));  //8x16  least 8x8
      __m128i tref_r2 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+2+16));  //8x16  least 8x8
      __m128i tref_r3 = _mm_loadl_epi64((__m128i *) (ref+y+iIdx+2+24));  //8x16  least 8x8

      do {
        switch(lidx) {
          case 0 : vidx_x = _mm_set_epi16(0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01); break;
          case 1 : vidx_x = _mm_set_epi16(0x10, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09);; break;
          case 2 : vidx_x = _mm_set_epi16(0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11); break;
          case 3 : vidx_x = _mm_set_epi16(0x20, 0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19); break;
        }
        __m128i vidx_lx = _mm_mullo_epi16(vidx_x, _mm_set1_epi16(intraPredAngle));
        __m128i viIdx   = vshrn16(vidx_lx, 5); //least 64bit 8x8 avaliable
                viIdx   = _mm_sub_epi8(viIdx, _mm_set1_epi8(iIdx));

        __m128i viFact  = _mm_and_si128(vidx_lx, _mm_set1_epi16(31));  //16x8
                viFact  = _mm_shuffle_epi8 (viFact, *(__m128i*) mask8_16_even_odd);  //8x8
                viFact  = _mm_unpacklo_epi8(viFact, _mm_setzero_si128()); //16x8
        __m128i vsFact  = _mm_sub_epi16(_mm_set1_epi16(32), viFact); //16x8

        
        __m128i vref_l  = vtbl4(tref_l0, tref_l1, tref_l2, tref_l3, viIdx);  //8x8
                vref_l  = _mm_unpacklo_epi8(vref_l, _mm_setzero_si128()); //16x8
        __m128i vref_r  = vtbl4(tref_r0, tref_r1, tref_r2, tref_r3, viIdx);
                vref_r  = _mm_unpacklo_epi8(vref_r, _mm_setzero_si128()); //16x8

        __m128i weight_a =_mm_mullo_epi16 (vref_l,vsFact);
        __m128i weight_s = _mm_mullo_epi16 (vref_r,viFact);
                weight_s = _mm_add_epi16(weight_a, weight_s);

        __m128i val = vrshrn16(weight_s, 5);
        _mm_storel_epi64((__m128i *) (dst+y*dstStride+lidx*8), val);

      } while (++lidx < (nT/8)) ;

    } while (++y < nT) ;
  }

  if (intraPredMode==10 && cIdx==0 && nT<32 && !disableIntraBoundaryFilter) {  // DIFF 26->10
    for (int x=0;x<nT;x++) { // DIFF (x<->y)
      dst[x] = Clip_BitDepth(border[-1] + ((border[1+x] - border[0])>>1), bit_depth); // DIFF (x<->y && neg)
    }
  }

  return ;
}


// angle 2 ~ 9
void intra_prediction_angular_2_9_sse4(uint8_t* dst, int dstStride,
                                       int bit_depth, bool disableIntraBoundaryFilter,
                                       int xB0,int yB0,
                                       IntraPredMode intraPredMode,
                                       int nT,int cIdx,
                                       uint8_t * border)
{
  int intraPredAngle = intraPredAngle_table[intraPredMode];
  static const int8_t mask_rev_e8[16] = {7,6,5,4,3,2,1,0, 15,14,13,12,11,10,9, 8};
  static const int8_t mask8_16_even_odd[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7,  9, 11, 13, 15 };
  // 4x4
    if(nT == 4) 
    {
        uint8_t    iIdx   = ((0+1)*intraPredAngle)>>5;  // iIdx < nT
        const __m128i    vidx_x = _mm_set_epi16(0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01);  //16x8
        __m128i    vidx_lx = _mm_mullo_epi16(vidx_x, _mm_set1_epi16(intraPredAngle)); //16x8
        __m128i    viIdx = vshrn16(vidx_lx,5) ;

        __m128i    viFact = _mm_and_si128(vidx_lx, _mm_set1_epi16(31));
                   viFact = _mm_shuffle_epi8 (viFact, *(__m128i*) mask8_16_even_odd);
                   viFact = _mm_unpacklo_epi8(viFact, _mm_setzero_si128()); //16bits
        __m128i    vsFact = _mm_sub_epi16(_mm_set1_epi16(32), viFact);

        for (int y = 0; y < nT; y++)
        {
                __m128i tref_l = _mm_loadl_epi64((__m128i *) (border-y-1-7));  //least 8x8 avaliable
                        tref_l = _mm_shuffle_epi8 (tref_l, *(__m128i*)  mask_rev_e8); //least 8x8 avaliable

                __m128i vref_l = vtbl1(tref_l,viIdx); //least 8x8 avaliable
                        vref_l = _mm_unpacklo_epi8(vref_l, _mm_setzero_si128()); //16bits
                __m128i tref_r = _mm_loadl_epi64((__m128i *) (border-y-2-7)); 
                        tref_r = _mm_shuffle_epi8 (tref_r, *(__m128i*)  mask_rev_e8);
                __m128i vref_r = vtbl1(tref_r,viIdx);
                        vref_r = _mm_unpacklo_epi8(vref_r, _mm_setzero_si128()); //16bits

                __m128i weight_a = _mm_mullo_epi16 (vref_l,vsFact);
                __m128i tmp = _mm_mullo_epi16(vref_r,viFact);
                __m128i weight_s = _mm_add_epi16 (tmp, weight_a);

                __m128i val = vrshrn16(weight_s ,5);
                *((uint32_t *)(dst+y*dstStride)) = _mm_cvtsi128_si32(val);
        }
    } 
    else 
    {
        for (int y = 0; y < nT; ++y)
        {
                for (int lidx = 0; lidx < (nT/8); ++lidx)
                {
                        uint8_t    iIdx   = ((lidx*8+1)*intraPredAngle)>>5;  // iIdx < nT
                        __m128i    vidx_x = _mm_setzero_si128();
                        switch(lidx) 
                        {
                                case 0 : 
                                        vidx_x = _mm_set_epi16(0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01);
                                        break;
                                case 1 : 
                                        vidx_x = _mm_set_epi16(0x10, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09);
                                        break;
                                case 2 : 
                                        vidx_x = _mm_set_epi16(0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11);
                                        break;
                                case 3 : 
                                        vidx_x = _mm_set_epi16(0x20, 0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19);
                                        break;
                        }

                        __m128i vidx_lx = _mm_mullo_epi16(vidx_x, _mm_set1_epi16(intraPredAngle));
                        __m128i viIdx   = vshrn16(vidx_lx, 5); //least 64bit 8x8 avaliable
                                viIdx   = _mm_sub_epi8(viIdx, _mm_set1_epi8(iIdx));
                        __m128i viFact  = _mm_and_si128(vidx_lx, _mm_set1_epi16(31));  //16x8
                                viFact  = _mm_shuffle_epi8 (viFact, *(__m128i*) mask8_16_even_odd);  //8x8
                                viFact  = _mm_unpacklo_epi8(viFact, _mm_setzero_si128()); //16x8
                        __m128i vsFact  = _mm_sub_epi16(_mm_set1_epi16(32), viFact); //16x8
                        __m128i tref_l0 = _mm_loadl_epi64((__m128i *) (border-y-iIdx-1-7));
                                tref_l0 = _mm_shuffle_epi8 (tref_l0, *(__m128i*)  mask_rev_e8); //8x8
                        __m128i tref_l1 = _mm_loadl_epi64((__m128i *) (border-y-iIdx-1-15));
                                tref_l1 = _mm_shuffle_epi8 (tref_l1, *(__m128i*)  mask_rev_e8); //8x8


                        __m128i tref_r0 = _mm_loadl_epi64((__m128i *) (border-y-iIdx-2-7));
                                tref_r0 = _mm_shuffle_epi8 (tref_r0, *(__m128i*)  mask_rev_e8); //8x8
                        __m128i tref_r1 = _mm_loadl_epi64((__m128i *) (border-y-iIdx-2-15));
                                tref_r1 = _mm_shuffle_epi8 (tref_r1, *(__m128i*)  mask_rev_e8); //8x8
                        __m128i vref_l  = vtbl2_u8(tref_l0, tref_l1, viIdx);  //8x8
                                vref_l  = _mm_unpacklo_epi8(vref_l, _mm_setzero_si128()); //16x8
                        __m128i vref_r  = vtbl2_u8(tref_r0, tref_r1, viIdx);
                                vref_r  = _mm_unpacklo_epi8(vref_r, _mm_setzero_si128()); //16x8
                        __m128i weight_a =_mm_mullo_epi16 (vref_l,vsFact);
                        __m128i weight_s = _mm_mullo_epi16 (vref_r,viFact);
                                weight_s = _mm_add_epi16(weight_a, weight_s);


                        __m128i val = vrshrn16(weight_s, 5);

                        _mm_storel_epi64((__m128i *) (dst+y*dstStride+lidx*8), val);
                } 
        } 
    }

  return ;
}
void intra_prediction_planar_8_sse4(uint8_t *_src, int _dstStride, int nT,int cIdx, uint8_t *border)
{
        uint64_t coef0 = 0x0403020104030201;
        uint64_t coef1 = 0x0001020300010203;
        uint64_t coef2 = 0x0202020203030303;
        uint64_t coef3 = 0x0000000001010101;
        uint64_t coef4 = 0x0202020201010101;
        uint64_t coef5 = 0x0404040403030303;

        uint64_t coef6 = 0x0807060504030201;
        if(nT == 4)
        {
                uint32_t   temp;
                memcpy(&temp, border+1, 4);
                __m128i border_tc = _mm_set1_epi32(temp); //32x2

                __m128i border_l0 = _mm_unpacklo_epi32(_mm_set1_epi8(border[-1]), _mm_set1_epi8(border[-2])); //8x8

                __m128i border_l2 = _mm_unpacklo_epi32(_mm_set1_epi8(border[-3]), _mm_set1_epi8(border[-4]));
                __m128i border_tr = _mm_set1_epi8(border[1+nT]);
                __m128i border_lb = _mm_set1_epi8(border[-1-nT]);

                __m128i coef_tr = _mm_set1_epi64x(coef0);
                __m128i coef_lc = _mm_set1_epi64x(coef1);
                __m128i coef_tc0 = _mm_set1_epi64x(coef2);
                __m128i coef_tc2 = _mm_set1_epi64x(coef3);
                __m128i coef_lb0 = _mm_set1_epi64x(coef4);
                __m128i coef_lb2 = _mm_set1_epi64x(coef5);

                __m128i sum_t0, sum_t2, sum_l0, sum_l2, sum_tl0, sum_tl2;

                        sum_t0 = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tr), _mm_cvtepu8_epi16(coef_tr));
                        sum_l0 = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_lb), _mm_cvtepu8_epi16(coef_lb0));
                __m128i tmp0   = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tc), _mm_cvtepu8_epi16(coef_tc0));
                        sum_t0 = _mm_add_epi16(tmp0, sum_t0);
                __m128i tmp1   = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_l0), _mm_cvtepu8_epi16(coef_lc));
                        sum_l0 = _mm_add_epi16(tmp1, sum_l0);
                        sum_tl0 = _mm_add_epi16(sum_t0, sum_l0);

                        sum_t2 = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tr), _mm_cvtepu8_epi16(coef_tr));
                        sum_l2 = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_lb), _mm_cvtepu8_epi16(coef_lb2));
                        tmp0   = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tc), _mm_cvtepu8_epi16(coef_tc2));
                        sum_t2 = _mm_add_epi16(tmp0, sum_t2);
                        tmp1   = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_l2), _mm_cvtepu8_epi16(coef_lc));
                        sum_l2 = _mm_add_epi16(tmp1, sum_l2);
                        sum_tl2 = _mm_add_epi16(sum_t2, sum_l2);

                __m128i shf_tl0  = vrshrn16(sum_tl0, 3);
                __m128i shf_tl2  = vrshrn16(sum_tl2, 3);

                *((uint32_t *) _src                 ) = _mm_cvtsi128_si32(shf_tl0   );
                *((uint32_t *)(_src +     _dstStride)) = _mm_extract_epi32(shf_tl0, 1);
                *((uint32_t *)(_src + 2 * _dstStride)) = _mm_extract_epi32(shf_tl2, 2);
                *((uint32_t *)(_src + 3 * _dstStride)) = _mm_extract_epi32(shf_tl2, 3);
        }
        else
        {
                __m128i border_tr = _mm_set1_epi8(border[1+nT]); //8x8
                __m128i border_lb = _mm_set1_epi8(border[-1-nT]); //8x8
                __m128i base = _mm_set1_epi64x(coef6);

                for(int y=0; y<nT; y++) 
                {
                        __m128i coef_lb = _mm_set1_epi8(y+1); //8x8
                                coef_lb = _mm_cvtepu8_epi16(coef_lb);//16x8
                        __m128i coef_tc = _mm_set1_epi8(nT-1-y); //8x8
                                coef_tc = _mm_cvtepu8_epi16(coef_tc);//16x8
                        __m128i border_lc = _mm_set1_epi8(border[-1-y]); //8x8

                        for(int x=0; x<nT; x+=8) 
                        {
                                __m128i border_tc = _mm_loadl_epi64((__m128i *) (border+1+x));  //8x8
                                __m128i coef_tr = _mm_add_epi8(base, _mm_set1_epi8(x));  //8x8
                                        coef_tr = _mm_cvtepu8_epi16(coef_tr); //16x8
                                __m128i coef_lc = _mm_sub_epi16(_mm_set1_epi16(nT), coef_tr);  //16x8

                                __m128i sum_tc, sum_lc, sum_tr, sum_lb, sum_t,sum_l, sum;

                                sum_tc = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tc), coef_tc);
                                sum_tr = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tr), coef_tr);
                                sum_lc = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_lc), coef_lc);
                                sum_lb = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_lb), coef_lb);

                                sum_t = _mm_add_epi16(sum_tc, sum_tr);
                                sum_l = _mm_add_epi16(sum_lc, sum_lb);
                                sum = _mm_add_epi16(sum_t, sum_l);

                                __m128i  val;
                                switch(nT) 
                                {
                                        case 8 :
                                                val = vrshrn16(sum, 4);
                                                break;
                                        case 16 :
                                                val = vrshrn16(sum, 5);
                                                break;
                                        case 32 :
                                                val = vrshrn16(sum, 6);
                                                break;
                                }
                                _mm_storel_epi64((__m128i *) (_src+y*_dstStride+x), val);

                        }
                }
            
        }
}


LIBDE265_INLINE __m128i vrshrq16(__m128i a, int b) // VRSHR.S16 q0,q0,#16
{
    __m128i maskb, r;
    maskb =  _mm_slli_epi16(a, (16 - b)); //to get rounding (b-1)th bit
    maskb = _mm_srli_epi16(maskb, 15); //1 or 0
    r = _mm_srli_epi16 (a, b);
    return _mm_add_epi16 (r, maskb); //actual rounding
}


void intra_prediction_sample_filtering_sse4(const seq_parameter_set& sps,
                                            uint8_t* p,
                                            int nT, int cIdx,
                                            enum IntraPredMode intraPredMode)
{
  int filterFlag;
  static const int8_t mask8_16_even_odd[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7,  9, 11, 13, 15 };
  static const int8_t mask_rev_e8[16] = {7,6,5,4,3,2,1,0, 15,14,13,12,11,10,9, 8};

  //printf("filtering, mode: %d\n",intraPredMode);

  if (intraPredMode==INTRA_DC || nT==4) {
    filterFlag = 0;
  } else {
    // int-cast below prevents a typing problem that leads to wrong results when abs_value is a macro
    int minDistVerHor = libde265_min( abs_value((int)intraPredMode-26),
                                      abs_value((int)intraPredMode-10) );

    //printf("mindist: %d\n",minDistVerHor);

    switch (nT) {
    case 8:  filterFlag = (minDistVerHor>7) ? 1 : 0; break;
    case 16: filterFlag = (minDistVerHor>1) ? 1 : 0; break;
    case 32: filterFlag = (minDistVerHor>0) ? 1 : 0; break;
      // there is no official 64x64 TB block, but we call this for some intra-pred mode algorithms
      // on the whole CB (2Nx2N mode for the whole CTB)
    case 64: filterFlag = 0; break;
    default: filterFlag = -1; assert(false); break; // should never happen
    }
  }


  if (filterFlag) {
    int biIntFlag = (sps.strong_intra_smoothing_enable_flag &&
                     cIdx==0 &&
                     nT==32 &&
                     abs_value(p[0]+p[ 64]-2*p[ 32]) < (1<<(sps.bit_depth_luma-5)) &&
                     abs_value(p[0]+p[-64]-2*p[-32]) < (1<<(sps.bit_depth_luma-5)))
      ? 1 : 0;

    uint8_t  pF_mem[4*32+1];
    uint8_t* pF = &pF_mem[2*32];
    uint8_t  p0 = p[0];

    if (biIntFlag) {
      pF[-2*nT] = p[-2*nT];
      pF[ 2*nT] = p[ 2*nT];
      pF[    0] = p[    0];

//      for (int i=1;i<=63;i++) {
//        pF[-i] = p[0] + ((i*(p[-64]-p[0])+32)>>6);
//        pF[ i] = p[0] + ((i*(p[ 64]-p[0])+32)>>6);
//      }
      // neon 
      int16_t pFp = p[-64] - p[0];
      int16_t pFn = p[ 64] - p[0];
      __m128i vpFp ;
      __m128i vpFn ;
      int icnt = 0;
      __m128i vi = _mm_set_epi16 (0x00007, 0x0006, 0x0005, 0x0004, 0x0003, 0x0002, 0x0001, 0x0000);
      //__m128i vi = vcombine_s16(vcreate_s16(0x0003000200010000),vcreate_s16(0x0007000600050004));
      do {
        vpFp = _mm_set1_epi16(pFp);
        vpFp = _mm_mullo_epi16(vpFp, vi);
        vpFp = _mm_add_epi16(_mm_set1_epi16(32), vpFp);
        vpFp = _mm_srai_epi16(vpFp,6);
        vpFp = _mm_add_epi16(vpFp,_mm_set1_epi16(p0));

        vpFp = _mm_shuffle_epi8 (vpFp, *(__m128i*) mask8_16_even_odd); 
        vpFp = _mm_shuffle_epi8 (vpFp, *(__m128i*)  mask_rev_e8);
        _mm_storel_epi64((__m128i *) (pF-icnt-7), vpFp);

        vpFn = _mm_set1_epi16(pFn);     
        vpFn = _mm_mullo_epi16(vpFn, vi);
        vpFn = _mm_add_epi16(_mm_set1_epi16(32), vpFn);
        vpFn = _mm_srai_epi16(vpFn,6);
        vpFn = _mm_add_epi16(vpFn,_mm_set1_epi16(p0));

        vpFn = _mm_shuffle_epi8 (vpFn, *(__m128i*) mask8_16_even_odd); //use 64 low bits only
        _mm_storel_epi64((__m128i *) (pF+icnt), vpFn);

        vi = _mm_add_epi16(vi, _mm_set1_epi16(8));
        icnt +=8;
      } while (icnt < 64);
      pF[0]  = p0;

    } else {
      pF[-2*nT] = p[-2*nT];
      pF[ 2*nT] = p[ 2*nT];
//      for (int i=-(2*nT-1) ; i<=2*nT-1 ; i++)
//        {
//          pF[i] = (p[i+1] + 2*p[i] + p[i-1] + 2) >> 2;
//        }
      int y = 0 ;
      uint8_t   *pnew = p-(2*nT-1)  ;
      uint8_t   *pFnew= pF-(2*nT-1) ;
      do {
        __m128i pl    = _mm_loadl_epi64((__m128i *) (pnew-1));
                pl    = _mm_unpacklo_epi8(pl, _mm_setzero_si128());
        __m128i pm    = _mm_loadl_epi64((__m128i *) (pnew));
                pm    = _mm_unpacklo_epi8(pm, _mm_setzero_si128());
        __m128i pr    = _mm_loadl_epi64((__m128i *) (pnew+1));
                pr    = _mm_unpacklo_epi8(pr, _mm_setzero_si128());

                pl    = _mm_add_epi16(pl, pm);
                pm    = _mm_add_epi16(pm, pr);
        __m128i ps    = _mm_add_epi16(pl, pm);

        __m128i po    = vrshrq16(ps,2);
                po    = _mm_packus_epi16 (po,po);

        _mm_storel_epi64((__m128i *) (pFnew), po);

        pnew += ((y==(nT*2/8-1))?7:8); // when pnew = p[0], +7
        pFnew+= ((y==(nT*2/8-1))?7:8); 
      } while ( ++y < nT*2*2/8);
    }

    // copy back to original array

    memcpy(p-2*nT, pF-2*nT, (4*nT+1) * sizeof(uint8_t));
  }
  else {
    // do nothing ?
  }


  logtrace(LogIntraPred,"post filtering: ");
  print_border(p,NULL,nT);
  logtrace(LogIntraPred,"\n");
}
#endif

#if HAVE_AVX2

LIBDE265_INLINE int hsum_int_avx(__m256i v) {
    __m128i vLow = _mm256_castsi256_si128(v);
    __m128i vHigh = _mm256_extracti128_si256(v, 1);
     vLow = _mm_add_epi32(vLow, vHigh);
     vLow = _mm_hadd_epi16(vLow, vLow);
     vLow = _mm_hadd_epi16(vLow, vLow);
     vLow = _mm_unpacklo_epi16(vLow, _mm_setzero_si128());
     vLow = _mm_hadd_epi32(vLow, vLow);

    return _mm_cvtsi128_si32(vLow);
}

void intra_prediction_DC_32_8_avx2(uint8_t *_dst, int dstStride, int nT, int cIdx, /* const */ uint8_t *border)
{
    uint8_t *dst = _dst;
    uint8_t* top = border+1;
    uint8_t* left = border-nT;

    //Load Border
    __m256i val[4];
    __m256i val_t  = _mm256_lddqu_si256((__m256i *) (top)); 
            val[0] = _mm256_unpacklo_epi8(val_t, _mm256_setzero_si256());
            val[1] = _mm256_unpackhi_epi8(val_t, _mm256_setzero_si256());

    __m256i val_l  = _mm256_lddqu_si256((__m256i *) (left)); 
            val[2] = _mm256_unpacklo_epi8(val_l, _mm256_setzero_si256());
            val[3] = _mm256_unpackhi_epi8(val_l, _mm256_setzero_si256());

    int sum = 0;
    for (int i = 0; i < 4; i++)
    {
        sum += hsum_int_avx(val[i]);
    }

    int32_t DcVal = (sum + nT) >> 6;

    //cpy dc to all pixels
    __m256i value = _mm256_set1_epi8(DcVal);
    for (int i = 0; i < nT; i++)
    {
        _mm256_storeu_si256((__m256i *) dst, value);
        dst +=dstStride ;
    }

}

LIBDE265_INLINE __m256i vrshrn16_avx2(__m256i a, int b) // VRSHRN.I16 d0,q0,#8
{
    __m256i mask, r16;
    mask = _mm256_set1_epi16(0xff);

    __m256i maskb =  _mm256_slli_epi16(a, (16 - b));
    maskb = _mm256_srli_epi16(maskb, 15); 
    r16  = _mm256_srai_epi16(a,b); //after right shift b>=1 unsigned var fits into signed range, so we could use _mm_packus_epi16 (signed 16 to unsigned 8)
    r16 = _mm256_add_epi16 (r16, maskb);
    r16 = _mm256_and_si256(r16, mask); //to avoid saturation
    r16 = _mm256_packus_epi16 (r16,r16); //saturate and  narrow, use low 64 bits only
    return (r16);
}

void intra_prediction_planar_8_avx2(uint8_t *_src, int _dstStride, int nT,int cIdx, uint8_t *border)
{
        uint64_t coef0 = 0x0403020104030201;
        uint64_t coef1 = 0x0001020300010203;
        uint64_t coef2 = 0x0202020203030303;
        uint64_t coef3 = 0x0000000001010101;
        uint64_t coef4 = 0x0202020201010101;
        uint64_t coef5 = 0x0404040403030303;

        uint64_t coef6 = 0x0807060504030201;
        uint64_t coef7 = 0x100f0e0d0c0b0a09;
        if(nT == 4)
        {
                uint32_t   temp;
                memcpy(&temp, border+1, 4);
                __m128i border_tc = _mm_set1_epi32(temp); //32x2

                __m128i border_l0 = _mm_unpacklo_epi32(_mm_set1_epi8(border[-1]), _mm_set1_epi8(border[-2])); //8x8

                __m128i border_l2 = _mm_unpacklo_epi32(_mm_set1_epi8(border[-3]), _mm_set1_epi8(border[-4]));
                __m128i border_tr = _mm_set1_epi8(border[1+nT]);
                __m128i border_lb = _mm_set1_epi8(border[-1-nT]);

                __m128i coef_tr = _mm_set1_epi64x(coef0);
                __m128i coef_lc = _mm_set1_epi64x(coef1);
                __m128i coef_tc0 = _mm_set1_epi64x(coef2);
                __m128i coef_tc2 = _mm_set1_epi64x(coef3);
                __m128i coef_lb0 = _mm_set1_epi64x(coef4);
                __m128i coef_lb2 = _mm_set1_epi64x(coef5);

                __m128i sum_t0, sum_t2, sum_l0, sum_l2, sum_tl0, sum_tl2;

                        sum_t0 = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tr), _mm_cvtepu8_epi16(coef_tr));
                        sum_l0 = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_lb), _mm_cvtepu8_epi16(coef_lb0));
                __m128i tmp0   = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tc), _mm_cvtepu8_epi16(coef_tc0));
                        sum_t0 = _mm_add_epi16(tmp0, sum_t0);
                __m128i tmp1   = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_l0), _mm_cvtepu8_epi16(coef_lc));
                        sum_l0 = _mm_add_epi16(tmp1, sum_l0);
                        sum_tl0 = _mm_add_epi16(sum_t0, sum_l0);

                        sum_t2 = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tr), _mm_cvtepu8_epi16(coef_tr));
                        sum_l2 = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_lb), _mm_cvtepu8_epi16(coef_lb2));
                        tmp0   = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tc), _mm_cvtepu8_epi16(coef_tc2));
                        sum_t2 = _mm_add_epi16(tmp0, sum_t2);
                        tmp1   = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_l2), _mm_cvtepu8_epi16(coef_lc));
                        sum_l2 = _mm_add_epi16(tmp1, sum_l2);
                        sum_tl2 = _mm_add_epi16(sum_t2, sum_l2);

                __m128i shf_tl0  = vrshrn16(sum_tl0, 3);
                __m128i shf_tl2  = vrshrn16(sum_tl2, 3);

                *((uint32_t *) _src                 ) = _mm_cvtsi128_si32(shf_tl0   );
                *((uint32_t *)(_src +     _dstStride)) = _mm_extract_epi32(shf_tl0, 1);
                *((uint32_t *)(_src + 2 * _dstStride)) = _mm_extract_epi32(shf_tl2, 2);
                *((uint32_t *)(_src + 3 * _dstStride)) = _mm_extract_epi32(shf_tl2, 3);
        }
        else if(nT == 8)
        {
                __m128i border_tr = _mm_set1_epi8(border[1+nT]); //8x8
                __m128i border_lb = _mm_set1_epi8(border[-1-nT]); //8x8
                __m128i base = _mm_set1_epi64x(coef6);

                for(int y=0; y<nT; y++) 
                {
                        __m128i coef_lb = _mm_set1_epi8(y+1); //8x8
                                coef_lb = _mm_cvtepu8_epi16(coef_lb);//16x8
                        __m128i coef_tc = _mm_set1_epi8(nT-1-y); //8x8
                                coef_tc = _mm_cvtepu8_epi16(coef_tc);//16x8
                        __m128i border_lc = _mm_set1_epi8(border[-1-y]); //8x8

                        for(int x=0; x<nT; x+=8) 
                        {
                                __m128i border_tc = _mm_loadl_epi64((__m128i *) (border+1+x));  //8x8
                                __m128i coef_tr = _mm_add_epi8(base, _mm_set1_epi8(x));  //8x8
                                        coef_tr = _mm_cvtepu8_epi16(coef_tr); //16x8
                                __m128i coef_lc = _mm_sub_epi16(_mm_set1_epi16(nT), coef_tr);  //16x8

                                __m128i sum_tc, sum_lc, sum_tr, sum_lb, sum_t,sum_l, sum;

                                sum_tc = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tc), coef_tc);
                                sum_tr = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_tr), coef_tr);
                                sum_lc = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_lc), coef_lc);
                                sum_lb = _mm_mullo_epi16(_mm_cvtepu8_epi16(border_lb), coef_lb);

                                sum_t = _mm_add_epi16(sum_tc, sum_tr);
                                sum_l = _mm_add_epi16(sum_lc, sum_lb);
                                sum = _mm_add_epi16(sum_t, sum_l);

                                __m128i  val;
                                switch(nT) 
                                {
                                        case 8 :
                                                val = vrshrn16(sum, 4);
                                                break;
                                        case 16 :
                                                val = vrshrn16(sum, 5);
                                                break;
                                        case 32 :
                                                val = vrshrn16(sum, 6);
                                                break;
                                }
                                _mm_storel_epi64((__m128i *) (_src+y*_dstStride+x), val);

                        }
                }
        }
        else
        {
                __m128i border_tr = _mm_set1_epi8(border[1+nT]); //8x16
                __m128i border_lb = _mm_set1_epi8(border[-1-nT]); //8x16
                __m128i base = _mm_set_epi64x(coef7, coef6); //8x16

                for(int y=0; y<nT; y++) 
                {
                        __m128i coef_lb0 = _mm_set1_epi8(y+1); //8x16
                        __m256i coef_lb = _mm256_cvtepu8_epi16(coef_lb0);//16x16
                        __m128i coef_tc0 = _mm_set1_epi8(nT-1-y); //8x16
                        __m256i coef_tc = _mm256_cvtepu8_epi16(coef_tc0);//16x16
                        __m128i border_lc = _mm_set1_epi8(border[-1-y]); //8x16

                        for(int x=0; x<nT; x+=16) 
                        {

                                __m128i border_tc = _mm_lddqu_si128((__m128i *) (border+1+x)); //8x16
                                __m128i coef_tr0 = _mm_add_epi8(base, _mm_set1_epi8(x));  //8x16
                                __m256i coef_tr = _mm256_cvtepu8_epi16(coef_tr0); //16x16
                                __m256i coef_lc = _mm256_sub_epi16(_mm256_set1_epi16(nT), coef_tr);  //16x16

                                __m256i sum_tc, sum_lc, sum_tr, sum_lb, sum_t,sum_l, sum; //16x16

                                sum_tc = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(border_tc), coef_tc);
                                sum_tr = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(border_tr), coef_tr);
                                sum_lc = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(border_lc), coef_lc);
                                sum_lb = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(border_lb), coef_lb);

                                sum_t = _mm256_add_epi16(sum_tc, sum_tr);
                                sum_l = _mm256_add_epi16(sum_lc, sum_lb);
                                sum = _mm256_add_epi16(sum_t, sum_l);

                                __m256i  val;
                                switch(nT) 
                                {
                                        case 8 :
                                                val = vrshrn16_avx2(sum, 4);
                                                break;
                                        case 16 :
                                                val = vrshrn16_avx2(sum, 5);
                                                break;
                                        case 32 :
                                                val = vrshrn16_avx2(sum, 6);
                                                break;
                                }
                                _mm_storel_epi64((__m128i *) (_src+y*_dstStride+x), _mm256_extracti128_si256(val,0));
                                _mm_storel_epi64((__m128i *) (_src+y*_dstStride+x+8), _mm256_extracti128_si256(val,1));
                        }

                }
   
        }
}

#endif