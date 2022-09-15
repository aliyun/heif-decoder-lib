#ifndef LIBDE265_NEON_DBK_H
#define LIBDE265_NEON_DBK_H

void ff_hevc_loop_filter_luma_8_neon(uint8_t *_dst, bool vertical, ptrdiff_t _stride, int _beta, int* _tc, uint8_t *_no_p, uint8_t *_no_q, int bitdepth);

#ifdef OPT_4
void ff_hevc_loop_filter_chroma_8_neon(uint8_t *_dst, bool vertical, ptrdiff_t _stride, int* _tc, uint8_t *no_p, uint8_t *no_q, int bitdepth);
#else
void ff_hevc_loop_filter_chroma_8_neon(uint8_t *_dst, bool vertical, ptrdiff_t _stride, int tc, bool filterP, bool filterQ, int bitdepth);
#endif


void ff_hevc_sao_band_filter_8_neon(uint8_t *_dst, int stride_dst, /*const*/ uint8_t *_src, int stride_src, int cIdx, int ctuW, int ctuH, int saoLeftClass, const int8_t *saoOffsetVal, int bandShift, const int maxPixelValue);
void ff_hevc_sao_edge_filter_8_neon(uint8_t* out_ptr, int out_stride, uint8_t* in_ptr , int in_stride, int SaoEoClass, int8_t* saoOffsetVal, int* edges, int ctuW, int ctuH, const int maxPixelValue);
#endif