#ifndef X86_DBK_H
#define X86_DBK_H

#include <stddef.h>
#include <stdint.h>


#if HAVE_SSE4_1
void ff_hevc_loop_filter_luma_8_sse4(uint8_t *_dst, bool vertical, ptrdiff_t _stride, int _beta, int* _tc, uint8_t *_no_p, uint8_t *_no_q, int bitdepth);


void ff_hevc_loop_filter_chroma_8_sse4(uint8_t *_dst, bool vertical, ptrdiff_t _stride, int* _tc, uint8_t *no_p, uint8_t *no_q, int bitdepth);
#endif

#endif