#include "heif.h"
#include "heif_plugin.h"
#include "heif_colorconversion.h"
#include "heif_api_structs.h"
#include <cassert>
#include <cstring>

#include "ihevc_typedefs.h"
#include "iv.h"
#include "ivd.h"
//#include "ihevcd_api.h"
#include "ihevcd_cxa.h"
//#include "ihevcd_function_selector.h"
//#include "ithread.h"
#include <stdio.h>

using namespace heif;

//#define ADAPTIVE_TEST
#define ADAPTIVE_MAX_WD           8192
#define ADAPTIVE_MAX_HT           4096
#define STRLENGTH                 1000
#define MAX_DISP_BUFFERS          64
#define STRIDE                    0

#define DEFAULT_NUM_CORES         1

typedef struct
{
    UWORD32 u4_piclen_flag;
    UWORD32 u4_file_save_flag;
    UWORD32 u4_frame_info_enable;
    UWORD32 u4_chksum_save_flag;
    UWORD32 u4_max_frm_ts;
    IV_COLOR_FORMAT_T e_output_chroma_format;
    IVD_ARCH_T e_arch;
    IVD_SOC_T e_soc;
    UWORD32 dump_q_rd_idx;
    UWORD32 dump_q_wr_idx;
    WORD32  disp_q_wr_idx;
    WORD32  disp_q_rd_idx;

    void *cocodec_obj;
    UWORD32 share_disp_buf;
    UWORD32 num_disp_buf;
    UWORD32 b_pic_present;
    WORD32 i4_degrade_type;
    WORD32 i4_degrade_pics;
    UWORD32 u4_num_cores;
    UWORD32 disp_delay;
    WORD32 trace_enable;
    CHAR ac_trace_fname[STRLENGTH];
    CHAR ac_piclen_fname[STRLENGTH];
    CHAR ac_ip_fname[STRLENGTH];
    CHAR ac_op_fname[STRLENGTH];
    CHAR ac_qp_map_fname[STRLENGTH];
    CHAR ac_blk_type_map_fname[STRLENGTH];
    CHAR ac_op_chksum_fname[STRLENGTH];
    ivd_out_bufdesc_t s_disp_buffers[MAX_DISP_BUFFERS];
    iv_yuv_buf_t s_disp_frm_queue[MAX_DISP_BUFFERS];
    UWORD32 s_disp_frm_id_queue[MAX_DISP_BUFFERS];
    UWORD32 loopback;
    UWORD32 display;
    UWORD32 full_screen;
    UWORD32 fps;
    UWORD32 max_wd;
    UWORD32 max_ht;
    UWORD32 max_level;

    UWORD32 u4_strd;

    /* For signalling to display thread */
    UWORD32 u4_pic_wd;
    UWORD32 u4_pic_ht;

    /* For IOS diplay */
    WORD32 i4_screen_wd;
    WORD32 i4_screen_ht;

    //UWORD32 u4_output_present;
    WORD32  quit;
    WORD32  paused;


    void *pv_disp_ctx;
    void *display_thread_handle;
    WORD32 display_thread_created;
    volatile WORD32 display_init_done;
    volatile WORD32 display_deinit_flag;

    void* (*disp_init)(UWORD32, UWORD32, WORD32, WORD32, WORD32, WORD32, WORD32, WORD32 *, WORD32 *);
    void (*alloc_disp_buffers)(void *);
    void (*display_buffer)(void *, WORD32);
    void (*set_disp_buffers)(void *, WORD32, UWORD8 **, UWORD8 **, UWORD8 **);
    void (*disp_deinit)(void *);
    void (*disp_usleep)(UWORD32);
    IV_COLOR_FORMAT_T (*get_color_fmt)(void);
    UWORD32 (*get_stride)(void);
} vid_dec_ctx_t;

#define ivd_cxa_api_function   ihevcd_cxa_api_function

#ifdef _WIN32
/*****************************************************************************/
/* Function to print library calls                                           */
/*****************************************************************************/
/*****************************************************************************/
/*                                                                           */
/*  Function Name : memalign                                                 */
/*                                                                           */
/*  Description   : Returns malloc data. Ideally should return aligned memory*/
/*                  support alignment will be added later                    */
/*                                                                           */
/*  Inputs        : alignment                                                */
/*                  size                                                     */
/*  Globals       :                                                          */
/*  Processing    :                                                          */
/*                                                                           */
/*  Outputs       :                                                          */
/*  Returns       :                                                          */
/*                                                                           */
/*  Issues        :                                                          */
/*                                                                           */
/*  Revision History:                                                        */
/*                                                                           */
/*         DD MM YYYY   Author(s)       Changes                              */
/*         07 09 2012   100189          Initial Version                      */
/*                                                                           */
/*****************************************************************************/

void *ihevca_aligned_malloc(void *pv_ctxt, WORD32 alignment, WORD32 i4_size)
{
    (void)pv_ctxt;
    return (void *)_aligned_malloc(i4_size, alignment);
}

void ihevca_aligned_free(void *pv_ctxt, void *pv_buf)
{
    (void)pv_ctxt;
    _aligned_free(pv_buf);
    return;
}
#endif

#if IOS
void *ihevca_aligned_malloc(void *pv_ctxt, WORD32 alignment, WORD32 i4_size)
{
    (void)pv_ctxt;
    return malloc(i4_size);
}

void ihevca_aligned_free(void *pv_ctxt, void *pv_buf)
{
    (void)pv_ctxt;
    free(pv_buf);
    return;
}
#endif

#if (!defined(IOS)) && (!defined(_WIN32))
void *ihevca_aligned_malloc(void *pv_ctxt, WORD32 alignment, WORD32 i4_size)
{
    void *buf = NULL;
    (void)pv_ctxt;
    if (0 != posix_memalign(&buf, alignment, i4_size))
    {
        return NULL;
    }
    return buf;
}

void ihevca_aligned_free(void *pv_ctxt, void *pv_buf)
{
    (void)pv_ctxt;
    free(pv_buf);
    return;
}
#endif

struct libhevc_decoder
{
  iv_obj_t *codec_obj;
  ivd_out_bufdesc_t *ps_out_buf;
  UWORD8 *pu1_bs_buf = NULL;
  UWORD8 *pu2_bs_buf = NULL;
  UWORD32 u4_ip_buf_len;
  UWORD32 u4_ip_frm_ts;
  UWORD32 u4_op_frm_ts;
  UWORD32 u4_num_bytes_dec;
  UWORD32 file_pos;
  WORD32 total_bytes_comsumed;
  WORD32 u4_bytes_remaining = 0;
  bool strict_decoding = false;
};

static const char kEmptyString[] = "";
static vid_dec_ctx_t s_app_ctx;
static const int LIBHEVC_PLUGIN_PRIORITY = 50;

#define MAX_PLUGIN_NAME_LENGTH 80

static char plugin_name[MAX_PLUGIN_NAME_LENGTH];

static const char* libhevc_plugin_name()
{
  strcpy(plugin_name, "libhevc HEVC decoder");

  // const char* libde265_version = de265_get_version();

  // if (strlen(libde265_version) + 10 < MAX_PLUGIN_NAME_LENGTH) {
  //   strcat(plugin_name, ", version ");
  //   strcat(plugin_name, libde265_version);
  // }

  return plugin_name;
}

static void libhevc_init_plugin()
{
}

static void libhevc_deinit_plugin()
{
}

static int libhevc_does_support_format(enum heif_compression_format format)
{
  if (format == heif_compression_HEVC) {
    return LIBHEVC_PLUGIN_PRIORITY;
  }
  else {
    return 0;
  }
}

static struct heif_error libhevc_new_decoder(void** dec, int nthreads)
{
  WORD32 ret;
  struct libhevc_decoder* decoder = new libhevc_decoder();
  struct heif_error err = {heif_error_Ok, heif_suberror_Unspecified, kSuccess};

  //init
  decoder->u4_ip_frm_ts = 0;
  decoder->u4_num_bytes_dec = 0;
  decoder->file_pos = 0;
  decoder->total_bytes_comsumed = 0;
  decoder->u4_bytes_remaining = 0;

  s_app_ctx.share_disp_buf = 0;
  s_app_ctx.e_output_chroma_format = IV_YUV_420P;
  s_app_ctx.u4_num_cores = DEFAULT_NUM_CORES;
  s_app_ctx.e_arch = ARCH_ARMV8_GENERIC;
  s_app_ctx.e_soc = SOC_GENERIC;
  s_app_ctx.u4_max_frm_ts = 1;
  s_app_ctx.i4_degrade_type = 0;
  s_app_ctx.i4_degrade_pics = 0;

  /***********************************************************************/
  /*                      Create decoder instance                        */
  /***********************************************************************/
  decoder->ps_out_buf = (ivd_out_bufdesc_t *)malloc(sizeof(ivd_out_bufdesc_t));
  /*****************************************************************************/
  /*   API Call: Initialize the Decoder                                        */
  /*****************************************************************************/
  {
    ihevcd_cxa_create_ip_t s_create_ip;
    ihevcd_cxa_create_op_t s_create_op;
    //void *fxns = &ivd_cxa_api_function;
    s_create_ip.s_ivd_create_ip_t.e_cmd = IVD_CMD_CREATE;
    s_create_ip.s_ivd_create_ip_t.u4_share_disp_buf = s_app_ctx.share_disp_buf;
    s_create_ip.s_ivd_create_ip_t.e_output_format = (IV_COLOR_FORMAT_T)s_app_ctx.e_output_chroma_format;
    s_create_ip.s_ivd_create_ip_t.pf_aligned_alloc = ihevca_aligned_malloc;
    s_create_ip.s_ivd_create_ip_t.pf_aligned_free = ihevca_aligned_free;
    s_create_ip.s_ivd_create_ip_t.pv_mem_ctxt = NULL;
    s_create_ip.s_ivd_create_ip_t.u4_size = sizeof(ihevcd_cxa_create_ip_t);
    s_create_op.s_ivd_create_op_t.u4_size = sizeof(ihevcd_cxa_create_op_t);
    s_create_ip.u4_enable_frame_info = s_app_ctx.u4_frame_info_enable;

    ret = ivd_cxa_api_function(NULL, (void *)&s_create_ip, (void *)&s_create_op);

    if(ret != IV_SUCCESS)
    {
      err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "IVD_CMD_CREATE Fail"};
      return err;
    }
    decoder->codec_obj = (iv_obj_t*)s_create_op.s_ivd_create_op_t.pv_handle;
    //codec_obj->pv_fxns = fxns;
    decoder->codec_obj->u4_size = sizeof(iv_obj_t);
    s_app_ctx.cocodec_obj = decoder->codec_obj;
  }

  /*************************************************************************/
  /* set num of cores                                                      */
  /*************************************************************************/
  {

    ihevcd_cxa_ctl_set_num_cores_ip_t s_ctl_set_cores_ip;
    ihevcd_cxa_ctl_set_num_cores_op_t s_ctl_set_cores_op;

    s_ctl_set_cores_ip.e_cmd = IVD_CMD_VIDEO_CTL;
    s_ctl_set_cores_ip.e_sub_cmd = (IVD_CONTROL_API_COMMAND_TYPE_T)IHEVCD_CXA_CMD_CTL_SET_NUM_CORES;
    s_ctl_set_cores_ip.u4_num_cores = s_app_ctx.u4_num_cores;
    s_ctl_set_cores_ip.u4_size = sizeof(ihevcd_cxa_ctl_set_num_cores_ip_t);
    s_ctl_set_cores_op.u4_size = sizeof(ihevcd_cxa_ctl_set_num_cores_op_t);

    ret = ivd_cxa_api_function((iv_obj_t *)decoder->codec_obj, (void *)&s_ctl_set_cores_ip,
                               (void *)&s_ctl_set_cores_op);
    if(ret != IV_SUCCESS)
    {
      err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "IVD_CMD_VIDEO_CTL Fail"};
      return err;
    }
  }
  /*************************************************************************/
  /* set processsor                                                        */
  /*************************************************************************/
  {
    ihevcd_cxa_ctl_set_processor_ip_t s_ctl_set_num_processor_ip;
    ihevcd_cxa_ctl_set_processor_op_t s_ctl_set_num_processor_op;

    s_ctl_set_num_processor_ip.e_cmd = IVD_CMD_VIDEO_CTL;
    s_ctl_set_num_processor_ip.e_sub_cmd = (IVD_CONTROL_API_COMMAND_TYPE_T)IHEVCD_CXA_CMD_CTL_SET_PROCESSOR;
    s_ctl_set_num_processor_ip.u4_arch = s_app_ctx.e_arch;
    s_ctl_set_num_processor_ip.u4_soc = s_app_ctx.e_soc;
    s_ctl_set_num_processor_ip.u4_size = sizeof(ihevcd_cxa_ctl_set_processor_ip_t);
    s_ctl_set_num_processor_op.u4_size = sizeof(ihevcd_cxa_ctl_set_processor_op_t);

    ret = ivd_cxa_api_function((iv_obj_t *)decoder->codec_obj, (void *)&s_ctl_set_num_processor_ip,
                               (void *)&s_ctl_set_num_processor_op);
    if(ret != IV_SUCCESS)
    {
      err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "IVD_CMD_VIDEO_CTL Fail"};
      return err;
    }
  }

  *dec = decoder;
  return err;

}

void flush_output(iv_obj_t *codec_obj,
                  vid_dec_ctx_t *ps_app_ctx,
                  ivd_out_bufdesc_t *ps_out_buf,
                  UWORD8 *pu1_bs_buf,
                  UWORD32 *pu4_op_frm_ts,
                  FILE *ps_op_file,
                  FILE *ps_qp_file,
                  FILE *ps_cu_type_file,
                  UWORD8 *pu1_qp_map_buf,
                  UWORD8 *pu1_blk_type_map_buf,
                  FILE *ps_op_chksum_file,
                  UWORD32 u4_ip_frm_ts,
                  UWORD32 u4_bytes_remaining)
{
    WORD32 ret;

    do
    {

        ivd_ctl_flush_ip_t s_ctl_ip;
        ivd_ctl_flush_op_t s_ctl_op;

        if(*pu4_op_frm_ts >= (ps_app_ctx->u4_max_frm_ts + ps_app_ctx->disp_delay))
            break;

        s_ctl_ip.e_cmd = IVD_CMD_VIDEO_CTL;
        s_ctl_ip.e_sub_cmd = IVD_CMD_CTL_FLUSH;
        s_ctl_ip.u4_size = sizeof(ivd_ctl_flush_ip_t);
        s_ctl_op.u4_size = sizeof(ivd_ctl_flush_op_t);
        ret = ivd_cxa_api_function((iv_obj_t *)codec_obj, (void *)&s_ctl_ip,
                                   (void *)&s_ctl_op);

        if(ret != IV_SUCCESS)
        {
            printf("Error in Setting the decoder in flush mode\n");
        }

        if(IV_SUCCESS == ret)
        {
            ihevcd_cxa_video_decode_ip_t s_hevcd_video_decode_ip = {};
            ihevcd_cxa_video_decode_op_t s_hevcd_video_decode_op = {};
            ivd_video_decode_ip_t *ps_video_decode_ip =
                &s_hevcd_video_decode_ip.s_ivd_video_decode_ip_t;
            ivd_video_decode_op_t *ps_video_decode_op =
                &s_hevcd_video_decode_op.s_ivd_video_decode_op_t;

            ps_video_decode_ip->e_cmd = IVD_CMD_VIDEO_DECODE;
            ps_video_decode_ip->u4_ts = u4_ip_frm_ts;
            ps_video_decode_ip->pv_stream_buffer = pu1_bs_buf;
            ps_video_decode_ip->u4_num_Bytes = u4_bytes_remaining;
            ps_video_decode_ip->u4_size = sizeof(ihevcd_cxa_video_decode_ip_t);
            ps_video_decode_ip->s_out_buffer.u4_min_out_buf_size[0] =
                            ps_out_buf->u4_min_out_buf_size[0];
            ps_video_decode_ip->s_out_buffer.u4_min_out_buf_size[1] =
                            ps_out_buf->u4_min_out_buf_size[1];
            ps_video_decode_ip->s_out_buffer.u4_min_out_buf_size[2] =
                            ps_out_buf->u4_min_out_buf_size[2];

            ps_video_decode_ip->s_out_buffer.pu1_bufs[0] =
                            ps_out_buf->pu1_bufs[0];
            ps_video_decode_ip->s_out_buffer.pu1_bufs[1] =
                            ps_out_buf->pu1_bufs[1];
            ps_video_decode_ip->s_out_buffer.pu1_bufs[2] =
                            ps_out_buf->pu1_bufs[2];
            ps_video_decode_ip->s_out_buffer.u4_num_bufs =
                            ps_out_buf->u4_num_bufs;

            ps_video_decode_op->u4_size = sizeof(ihevcd_cxa_video_decode_op_t);
            s_hevcd_video_decode_ip.pu1_8x8_blk_qp_map = pu1_qp_map_buf;
            s_hevcd_video_decode_ip.pu1_8x8_blk_type_map = pu1_blk_type_map_buf;
            s_hevcd_video_decode_ip.u4_8x8_blk_qp_map_size =
                (ADAPTIVE_MAX_HT * ADAPTIVE_MAX_WD) >> 6;
            s_hevcd_video_decode_ip.u4_8x8_blk_type_map_size =
                (ADAPTIVE_MAX_HT * ADAPTIVE_MAX_WD) >> 6;

            /*****************************************************************************/
            /*   API Call: Video Decode                                                  */
            /*****************************************************************************/
            ret = ivd_cxa_api_function((iv_obj_t *)codec_obj, (void *)&s_hevcd_video_decode_ip,
                                       (void *)&s_hevcd_video_decode_op);

            if(1 == ps_video_decode_op->u4_output_present)
            {
                // dump_output(ps_app_ctx, &(ps_video_decode_op->s_disp_frm_buf),
                //             &s_hevcd_video_decode_op, ps_video_decode_op->u4_disp_buf_id,
                //             ps_op_file, ps_qp_file, ps_cu_type_file, ps_op_chksum_file,
                //             *pu4_op_frm_ts, ps_app_ctx->u4_file_save_flag,
                //             ps_app_ctx->u4_chksum_save_flag, ps_app_ctx->u4_frame_info_enable);

                (*pu4_op_frm_ts)++;
            }
        }
    }while(IV_SUCCESS == ret);

}

IV_API_CALL_STATUS_T set_degrade(void *codec_obj, UWORD32 type, WORD32 pics)
{
    ihevcd_cxa_ctl_degrade_ip_t s_ctl_ip;
    ihevcd_cxa_ctl_degrade_op_t s_ctl_op;
    void *pv_api_ip, *pv_api_op;
    IV_API_CALL_STATUS_T e_dec_status;

    s_ctl_ip.u4_size = sizeof(ihevcd_cxa_ctl_degrade_ip_t);
    s_ctl_ip.i4_degrade_type = type;
    s_ctl_ip.i4_nondegrade_interval = 4;
    s_ctl_ip.i4_degrade_pics = pics;

    s_ctl_op.u4_size = sizeof(ihevcd_cxa_ctl_degrade_op_t);

    pv_api_ip = (void *)&s_ctl_ip;
    pv_api_op = (void *)&s_ctl_op;

    s_ctl_ip.e_cmd = IVD_CMD_VIDEO_CTL;
    s_ctl_ip.e_sub_cmd = (IVD_CONTROL_API_COMMAND_TYPE_T)IHEVCD_CXA_CMD_CTL_DEGRADE;

    e_dec_status = ivd_cxa_api_function((iv_obj_t *)codec_obj, pv_api_ip, pv_api_op);

    if(IV_SUCCESS != e_dec_status)
    {
        printf("Error in setting degrade level \n");
    }
    return (e_dec_status);

}

IV_API_CALL_STATUS_T release_disp_frame(void *codec_obj, UWORD32 buf_id)
{
    ivd_rel_display_frame_ip_t s_video_rel_disp_ip;
    ivd_rel_display_frame_op_t s_video_rel_disp_op;
    IV_API_CALL_STATUS_T e_dec_status;

    s_video_rel_disp_ip.e_cmd = IVD_CMD_REL_DISPLAY_FRAME;
    s_video_rel_disp_ip.u4_size = sizeof(ivd_rel_display_frame_ip_t);
    s_video_rel_disp_op.u4_size = sizeof(ivd_rel_display_frame_op_t);
    s_video_rel_disp_ip.u4_disp_buf_id = buf_id;

    e_dec_status = ivd_cxa_api_function((iv_obj_t *)codec_obj, (void *)&s_video_rel_disp_ip,
                                        (void *)&s_video_rel_disp_op);
    if(IV_SUCCESS != e_dec_status)
    {
        printf("Error in Release Disp frame\n");
    }

    return (e_dec_status);
}

static struct heif_error libhevc_push_data(void* decoder_raw, const void* data, size_t size)
{
  WORD32 ret;
  struct heif_error err = {heif_error_Ok, heif_suberror_Unspecified, kSuccess};
  struct libhevc_decoder* decoder = (struct libhevc_decoder*) decoder_raw;
  const uint8_t* cdata = (const uint8_t*) data;

  // flush_output(decoder->codec_obj, &s_app_ctx, decoder->ps_out_buf, decoder->pu1_bs_buf, &decoder->u4_op_frm_ts,
  //              NULL, NULL, NULL,
  //              NULL, NULL,
  //              NULL, decoder->u4_ip_frm_ts, decoder->u4_bytes_remaining);

  /*****************************************************************************/
  /*   Decode header to get width and height and buffer sizes                  */
  /*****************************************************************************/
  {
    ihevcd_cxa_video_decode_ip_t s_hevcd_video_decode_ip = {};
    ihevcd_cxa_video_decode_op_t s_hevcd_video_decode_op = {};
    ivd_video_decode_ip_t *ps_video_decode_ip = &s_hevcd_video_decode_ip.s_ivd_video_decode_ip_t;
    ivd_video_decode_op_t *ps_video_decode_op = &s_hevcd_video_decode_op.s_ivd_video_decode_op_t;
    {
      ihevcd_cxa_ctl_set_config_ip_t s_hevcd_ctl_ip = {};
      ihevcd_cxa_ctl_set_config_op_t s_hevcd_ctl_op = {};
      ivd_ctl_set_config_ip_t *ps_ctl_ip = &s_hevcd_ctl_ip.s_ivd_ctl_set_config_ip_t;
      ivd_ctl_set_config_op_t *ps_ctl_op = &s_hevcd_ctl_op.s_ivd_ctl_set_config_op_t;

      ps_ctl_ip->u4_disp_wd = STRIDE;
      if(1 == s_app_ctx.display)
          ps_ctl_ip->u4_disp_wd = s_app_ctx.get_stride();

      ps_ctl_ip->e_frm_skip_mode = IVD_SKIP_NONE;
      ps_ctl_ip->e_frm_out_mode = IVD_DISPLAY_FRAME_OUT;
      ps_ctl_ip->e_vid_dec_mode = IVD_DECODE_HEADER;
      ps_ctl_ip->e_cmd = IVD_CMD_VIDEO_CTL;
      ps_ctl_ip->e_sub_cmd = IVD_CMD_CTL_SETPARAMS;
      ps_ctl_ip->u4_size = sizeof(ihevcd_cxa_ctl_set_config_ip_t);

      ps_ctl_op->u4_size = sizeof(ihevcd_cxa_ctl_set_config_op_t);

      ret = ivd_cxa_api_function((iv_obj_t*)decoder->codec_obj, (void *)&s_hevcd_ctl_ip, (void *)&s_hevcd_ctl_op);
      if(ret != IV_SUCCESS)
      {
        err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "IVD_CMD_VIDEO_CTL Fail"};
        return err;
      }
    }

   decoder->u4_ip_buf_len = (size+255)/256*256; //256 * 1024;
   //decoder->pu1_bs_buf = (UWORD8 *)malloc(decoder->u4_ip_buf_len);
   decoder->pu2_bs_buf = (UWORD8 *)malloc(decoder->u4_ip_buf_len);

   if(decoder->pu2_bs_buf == NULL)
   {
      err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "new input buffer FAIL"};
      return err;
   }

   size_t ptr = 0;
   while (ptr < size) {
     if (4 > size - ptr) {
       err = {heif_error_Decoder_plugin_error, heif_suberror_End_of_data, kEmptyString};
       return err;
     }

     uint32_t nal_size = (cdata[ptr] << 24) | (cdata[ptr + 1] << 16) | (cdata[ptr + 2] << 8) | (cdata[ptr + 3]);
     uint32_t start_code = 0x01000000;

     memcpy(decoder->pu2_bs_buf+ptr, &start_code, 4);

     ptr += 4;

     if (nal_size > size - ptr) {
      err = {heif_error_Decoder_plugin_error, heif_suberror_End_of_data, kEmptyString};
      return err;
     }
     memcpy(decoder->pu2_bs_buf+ptr, cdata + ptr, nal_size);
     ptr += nal_size;
   }
   decoder->u4_bytes_remaining = ptr;
   if(0 == decoder->u4_bytes_remaining)
   {
      err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "input stream nodata FAIL"};
      return err;
   }

   do
   {
      ps_video_decode_ip->e_cmd = IVD_CMD_VIDEO_DECODE;
      ps_video_decode_ip->u4_ts = decoder->u4_ip_frm_ts;
      ps_video_decode_ip->pv_stream_buffer = decoder->pu2_bs_buf;
      ps_video_decode_ip->u4_num_Bytes = decoder->u4_bytes_remaining;
      ps_video_decode_ip->u4_size = sizeof(ihevcd_cxa_video_decode_ip_t);

      ps_video_decode_op->u4_size = sizeof(ihevcd_cxa_video_decode_op_t);
      s_hevcd_video_decode_ip.pu1_8x8_blk_qp_map = NULL; //pu1_qp_map_buf;
      s_hevcd_video_decode_ip.pu1_8x8_blk_type_map = NULL; //pu1_blk_type_map_buf;
      s_hevcd_video_decode_ip.u4_8x8_blk_qp_map_size = 0; //(ADAPTIVE_MAX_HT * ADAPTIVE_MAX_WD) >> 6;
      s_hevcd_video_decode_ip.u4_8x8_blk_type_map_size = 0; //(ADAPTIVE_MAX_HT * ADAPTIVE_MAX_WD) >> 6;

      /*****************************************************************************/
      /*   API Call: Header Decode                                                  */
      /*****************************************************************************/
      ret = ivd_cxa_api_function((iv_obj_t *)decoder->codec_obj, (void *)&s_hevcd_video_decode_ip, (void *)&s_hevcd_video_decode_op);

      if(ret != IV_SUCCESS)
      {
        //err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "header decode FAIL"};
        //return err;
      }
      decoder->u4_num_bytes_dec = ps_video_decode_op->u4_num_bytes_consumed;
      decoder->file_pos += decoder->u4_num_bytes_dec;
      decoder->total_bytes_comsumed += decoder->u4_num_bytes_dec;

   } while (ret != IV_SUCCESS);

   decoder->u4_bytes_remaining = decoder->u4_bytes_remaining - decoder->total_bytes_comsumed;
   s_app_ctx.u4_pic_wd = ps_video_decode_op->u4_pic_wd;
   s_app_ctx.u4_pic_ht = ps_video_decode_op->u4_pic_ht;

   //free(decoder->pu1_bs_buf );

   {
      ivd_ctl_getbufinfo_ip_t s_ctl_ip;
      ivd_ctl_getbufinfo_op_t s_ctl_op;
      WORD32 outlen = 0;

      s_ctl_ip.e_cmd = IVD_CMD_VIDEO_CTL;
      s_ctl_ip.e_sub_cmd = IVD_CMD_CTL_GETBUFINFO;
      s_ctl_ip.u4_size = sizeof(ivd_ctl_getbufinfo_ip_t);
      s_ctl_op.u4_size = sizeof(ivd_ctl_getbufinfo_op_t);
      ret = ivd_cxa_api_function((iv_obj_t*)decoder->codec_obj, (void *)&s_ctl_ip, (void *)&s_ctl_op);
      if(ret != IV_SUCCESS)
      {
        err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "IVD_CMD_CTL_GETBUFINFO FAIL"};
        return err;
      }
      decoder->u4_ip_buf_len = s_ctl_op.u4_min_in_buf_size[0];
#ifdef ADAPTIVE_TEST
      decoder->u4_ip_buf_len = ADAPTIVE_MAX_WD * ADAPTIVE_MAX_HT * 3 >> 1;
#endif
      decoder->pu1_bs_buf = (UWORD8 *)malloc(decoder->u4_ip_buf_len);
      if(decoder->pu1_bs_buf == NULL)
      {
        err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "Allocation input image buffer FAIL"};
        return err;
      }
#ifdef ADAPTIVE_TEST
     switch(s_app_ctx.e_output_chroma_format)
     {
        case IV_YUV_420P:
        {
          s_ctl_op.u4_min_out_buf_size[0] = ADAPTIVE_MAX_WD * ADAPTIVE_MAX_HT;
          s_ctl_op.u4_min_out_buf_size[1] = ADAPTIVE_MAX_WD * ADAPTIVE_MAX_HT >> 2;
          s_ctl_op.u4_min_out_buf_size[2] = ADAPTIVE_MAX_WD * ADAPTIVE_MAX_HT >> 2;
          break;
        }
        case IV_YUV_420SP_UV:
        case IV_YUV_420SP_VU:
        {
          s_ctl_op.u4_min_out_buf_size[0] = ADAPTIVE_MAX_WD * ADAPTIVE_MAX_HT;
          s_ctl_op.u4_min_out_buf_size[1] = ADAPTIVE_MAX_WD * ADAPTIVE_MAX_HT >> 1;
          s_ctl_op.u4_min_out_buf_size[2] = 0;
          break;
        }
        case IV_YUV_422ILE:
        {
          s_ctl_op.u4_min_out_buf_size[0] = ADAPTIVE_MAX_WD * ADAPTIVE_MAX_HT * 2;
          s_ctl_op.u4_min_out_buf_size[1] = 0;
          s_ctl_op.u4_min_out_buf_size[2] = 0;
          break;
        }
        case IV_RGBA_8888:
        {
          s_ctl_op.u4_min_out_buf_size[0] = ADAPTIVE_MAX_WD * ADAPTIVE_MAX_HT * 4;
          s_ctl_op.u4_min_out_buf_size[1] = 0;
          s_ctl_op.u4_min_out_buf_size[2] = 0;
          break;
        }
        case IV_RGB_565:
        {
          s_ctl_op.u4_min_out_buf_size[0] = ADAPTIVE_MAX_WD * ADAPTIVE_MAX_HT * 2;
          s_ctl_op.u4_min_out_buf_size[1] = 0;
          s_ctl_op.u4_min_out_buf_size[2] = 0;
          break;
        }
        default:
          break;
      }
#endif
      /* Allocate output buffer only if display buffers are not shared */
      /* Or if shared and output is 420P */
      if((0 == s_app_ctx.share_disp_buf) || (IV_YUV_420P == s_app_ctx.e_output_chroma_format))
      {
          decoder->ps_out_buf->u4_min_out_buf_size[0] = s_ctl_op.u4_min_out_buf_size[0];
          decoder->ps_out_buf->u4_min_out_buf_size[1] = s_ctl_op.u4_min_out_buf_size[1];
          decoder->ps_out_buf->u4_min_out_buf_size[2] = s_ctl_op.u4_min_out_buf_size[2];
          outlen = s_ctl_op.u4_min_out_buf_size[0];
          if(s_ctl_op.u4_min_num_out_bufs > 1)
              outlen += s_ctl_op.u4_min_out_buf_size[1];
          if(s_ctl_op.u4_min_num_out_bufs > 2)
              outlen += s_ctl_op.u4_min_out_buf_size[2];

          decoder->ps_out_buf->pu1_bufs[0] = (UWORD8 *)malloc(outlen);
          if(decoder->ps_out_buf->pu1_bufs[0] == NULL)
          {
            err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "Allocation input image buffer FAIL"};
            return err;
          }

          if(s_ctl_op.u4_min_num_out_bufs > 1)
              decoder->ps_out_buf->pu1_bufs[1] = decoder->ps_out_buf->pu1_bufs[0] + (s_ctl_op.u4_min_out_buf_size[0]);

          if(s_ctl_op.u4_min_num_out_bufs > 2)
              decoder->ps_out_buf->pu1_bufs[2] = decoder->ps_out_buf->pu1_bufs[1] + (s_ctl_op.u4_min_out_buf_size[1]);

          decoder->ps_out_buf->u4_num_bufs = s_ctl_op.u4_min_num_out_bufs;
      }
      /*****************************************************************************/
      /*   API Call: Allocate display buffers for display buffer shared case       */
      /*****************************************************************************/
      for(int i = 0; i < s_ctl_op.u4_num_disp_bufs; i++)
      {

        s_app_ctx.s_disp_buffers[i].u4_min_out_buf_size[0] = s_ctl_op.u4_min_out_buf_size[0];
        s_app_ctx.s_disp_buffers[i].u4_min_out_buf_size[1] = s_ctl_op.u4_min_out_buf_size[1];
        s_app_ctx.s_disp_buffers[i].u4_min_out_buf_size[2] = s_ctl_op.u4_min_out_buf_size[2];

        outlen = s_ctl_op.u4_min_out_buf_size[0];
        if(s_ctl_op.u4_min_num_out_bufs > 1)
            outlen += s_ctl_op.u4_min_out_buf_size[1];

        if(s_ctl_op.u4_min_num_out_bufs > 2)
            outlen += s_ctl_op.u4_min_out_buf_size[2];

        s_app_ctx.s_disp_buffers[i].pu1_bufs[0] = (UWORD8 *)malloc(outlen);

        if(s_app_ctx.s_disp_buffers[i].pu1_bufs[0] == NULL)
        {
            err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "Allocation display image buffer FAIL"};
            return err;
        }

        if(s_ctl_op.u4_min_num_out_bufs > 1)
            s_app_ctx.s_disp_buffers[i].pu1_bufs[1] = s_app_ctx.s_disp_buffers[i].pu1_bufs[0] + (s_ctl_op.u4_min_out_buf_size[0]);

        if(s_ctl_op.u4_min_num_out_bufs > 2)
            s_app_ctx.s_disp_buffers[i].pu1_bufs[2] = s_app_ctx.s_disp_buffers[i].pu1_bufs[1] + (s_ctl_op.u4_min_out_buf_size[1]);

        s_app_ctx.s_disp_buffers[i].u4_num_bufs = s_ctl_op.u4_min_num_out_bufs;
      }
      s_app_ctx.num_disp_buf = s_ctl_op.u4_num_disp_bufs;
    }
    // /* Create display thread and wait for the display buffers to be initialized */
    // if(1 == s_app_ctx.display)
    // {
    //         if(0 == s_app_ctx.display_thread_created)
    //         {
    //             s_app_ctx.display_init_done = 0;
    //             ithread_create(s_app_ctx.display_thread_handle, NULL,
    //                                                 (void *) &display_thread, (void *) &s_app_ctx);
    //             s_app_ctx.display_thread_created = 1;

    //             while(1)
    //             {
    //                 if(s_app_ctx.display_init_done)
    //                     break;

    //                 ithread_msleep(1);
    //             }
    //         }
    //         s_app_ctx.u4_strd = s_app_ctx.get_stride();
    //     }
    /*****************************************************************************/
    /*   API Call: Send the allocated display buffers to codec                   */
    /*****************************************************************************/
    {
      ivd_set_display_frame_ip_t s_set_display_frame_ip;
      ivd_set_display_frame_op_t s_set_display_frame_op;

      s_set_display_frame_ip.e_cmd = IVD_CMD_SET_DISPLAY_FRAME;
      s_set_display_frame_ip.u4_size = sizeof(ivd_set_display_frame_ip_t);
      s_set_display_frame_op.u4_size = sizeof(ivd_set_display_frame_op_t);

      s_set_display_frame_ip.num_disp_bufs = s_app_ctx.num_disp_buf;

      memcpy(&(s_set_display_frame_ip.s_disp_buffer), &(s_app_ctx.s_disp_buffers), s_app_ctx.num_disp_buf * sizeof(ivd_out_bufdesc_t));

      ret = ivd_cxa_api_function((iv_obj_t *)decoder->codec_obj, (void *)&s_set_display_frame_ip, (void *)&s_set_display_frame_op);

      if(IV_SUCCESS != ret)
      {
        err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "Set display buffer FAIL"};
        return err;
      }
    }
  }

  /*************************************************************************/
  /* Get frame dimensions for display buffers such as x_offset,y_offset    */
  /* etc. This information might be needed to set display buffer           */
  /* offsets in case of shared display buffer mode                         */
  /*************************************************************************/
  {

    ihevcd_cxa_ctl_get_frame_dimensions_ip_t s_ctl_get_frame_dimensions_ip;
    ihevcd_cxa_ctl_get_frame_dimensions_op_t s_ctl_get_frame_dimensions_op;

    s_ctl_get_frame_dimensions_ip.e_cmd = IVD_CMD_VIDEO_CTL;
    s_ctl_get_frame_dimensions_ip.e_sub_cmd = (IVD_CONTROL_API_COMMAND_TYPE_T)IHEVCD_CXA_CMD_CTL_GET_BUFFER_DIMENSIONS;
    s_ctl_get_frame_dimensions_ip.u4_size = sizeof(ihevcd_cxa_ctl_get_frame_dimensions_ip_t);
    s_ctl_get_frame_dimensions_op.u4_size = sizeof(ihevcd_cxa_ctl_get_frame_dimensions_op_t);

    ret = ivd_cxa_api_function((iv_obj_t *)decoder->codec_obj, (void *)&s_ctl_get_frame_dimensions_ip, (void *)&s_ctl_get_frame_dimensions_op);
    if(IV_SUCCESS != ret)
    {
      err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "Get buffer Dimensions FAIL"};
      return err;
    }
  }

  /*************************************************************************/
  /* Get VUI parameters                                                    */
  /*************************************************************************/
  {

    ihevcd_cxa_ctl_get_vui_params_ip_t s_ctl_get_vui_params_ip;
    ihevcd_cxa_ctl_get_vui_params_op_t s_ctl_get_vui_params_op;

    s_ctl_get_vui_params_ip.e_cmd = IVD_CMD_VIDEO_CTL;
    s_ctl_get_vui_params_ip.e_sub_cmd = (IVD_CONTROL_API_COMMAND_TYPE_T)IHEVCD_CXA_CMD_CTL_GET_VUI_PARAMS;
    s_ctl_get_vui_params_ip.u4_size = sizeof(ihevcd_cxa_ctl_get_vui_params_ip_t);
    s_ctl_get_vui_params_op.u4_size = sizeof(ihevcd_cxa_ctl_get_vui_params_op_t);

    ret = ivd_cxa_api_function((iv_obj_t *)decoder->codec_obj, (void *)&s_ctl_get_vui_params_ip,
                               (void *)&s_ctl_get_vui_params_op);
    if(IV_SUCCESS != ret)
    {
      err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "Get VUI params FAIL"};
      //return err;
    }

  }

  /*************************************************************************/
  /* Set the decoder in frame decode mode. It was set in header decode     */
  /* mode earlier                                                          */
  /*************************************************************************/
  {

    ivd_ctl_set_config_ip_t s_ctl_ip;
    ivd_ctl_set_config_op_t s_ctl_op;

    s_ctl_ip.u4_disp_wd = STRIDE;
    if(1 == s_app_ctx.display)
        s_ctl_ip.u4_disp_wd = s_app_ctx.get_stride();
    s_ctl_ip.e_frm_skip_mode = IVD_SKIP_NONE;

    s_ctl_ip.e_frm_out_mode = IVD_DISPLAY_FRAME_OUT;
    s_ctl_ip.e_vid_dec_mode = IVD_DECODE_FRAME;
    s_ctl_ip.e_cmd = IVD_CMD_VIDEO_CTL;
    s_ctl_ip.e_sub_cmd = IVD_CMD_CTL_SETPARAMS;
    s_ctl_ip.u4_size = sizeof(ivd_ctl_set_config_ip_t);

    s_ctl_op.u4_size = sizeof(ivd_ctl_set_config_op_t);

    ret = ivd_cxa_api_function((iv_obj_t *)decoder->codec_obj, (void *)&s_ctl_ip, (void *)&s_ctl_op);

    if(IV_SUCCESS != ret)
    {
      err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "Set Parameters FAIL"};
      //return err;
    }
  }
  /*************************************************************************/
  /* If required disable deblocking and sao at given level                 */
  /*************************************************************************/
  set_degrade(decoder->codec_obj, s_app_ctx.i4_degrade_type, s_app_ctx.i4_degrade_pics);
  // while(decoder->u4_op_frm_ts < (s_app_ctx.u4_max_frm_ts + s_app_ctx.disp_delay))
  // {
  //   if(decoder->u4_ip_frm_ts < s_app_ctx.num_disp_buf)
  //   {
  //     release_disp_frame(decoder->codec_obj, decoder->u4_ip_frm_ts);
  //   }
  //   /***********************************************************************/
  //   /*   Seek the file to start of current frame, this is equavelent of    */
  //   /*   having a parcer which tells the start of current frame            */
  //   /***********************************************************************/
  //   UWORD8 *pic_cur_ptr = NULL;
  //   {
  //     pic_cur_ptr = decoder->pu2_bs_buf + decoder->total_bytes_comsumed;
  //     if(u4_bytes_remaining == 0)
  //     {
  //       err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "Su4_bytes_remaining 0"};
  //       assert(0);
  //     }
  //     {
  //       ihevcd_cxa_video_decode_ip_t s_hevcd_video_decode_ip = {};
  //       ihevcd_cxa_video_decode_op_t s_hevcd_video_decode_op = {};
  //       ivd_video_decode_ip_t *ps_video_decode_ip = &s_hevcd_video_decode_ip.s_ivd_video_decode_ip_t;
  //       ivd_video_decode_op_t *ps_video_decode_op = &s_hevcd_video_decode_op.s_ivd_video_decode_op_t;
  //       ps_video_decode_ip->e_cmd = IVD_CMD_VIDEO_DECODE;
  //       ps_video_decode_ip->u4_ts = decoder->u4_ip_frm_ts;
  //       ps_video_decode_ip->pv_stream_buffer = pic_cur_ptr;
  //       ps_video_decode_ip->u4_num_Bytes = u4_bytes_remaining;
  //       ps_video_decode_ip->u4_size = sizeof(ihevcd_cxa_video_decode_ip_t);
  //       ps_video_decode_ip->s_out_buffer.u4_min_out_buf_size[0] = decoder->ps_out_buf->u4_min_out_buf_size[0];
  //       ps_video_decode_ip->s_out_buffer.u4_min_out_buf_size[1] = decoder->ps_out_buf->u4_min_out_buf_size[1];
  //       ps_video_decode_ip->s_out_buffer.u4_min_out_buf_size[2] = decoder->ps_out_buf->u4_min_out_buf_size[2];

  //       ps_video_decode_ip->s_out_buffer.pu1_bufs[0] = decoder->ps_out_buf->pu1_bufs[0];
  //       ps_video_decode_ip->s_out_buffer.pu1_bufs[1] = decoder->ps_out_buf->pu1_bufs[1];
  //       ps_video_decode_ip->s_out_buffer.pu1_bufs[2] = decoder->ps_out_buf->pu1_bufs[2];
  //       ps_video_decode_ip->s_out_buffer.u4_num_bufs = decoder->ps_out_buf->u4_num_bufs;

  //       ps_video_decode_op->u4_size = sizeof(ihevcd_cxa_video_decode_op_t);
  //       s_hevcd_video_decode_ip.pu1_8x8_blk_qp_map = NULL; //pu1_qp_map_buf;
  //       s_hevcd_video_decode_ip.pu1_8x8_blk_type_map = NULL; //pu1_blk_type_map_buf;
  //       s_hevcd_video_decode_ip.u4_8x8_blk_qp_map_size = (ADAPTIVE_MAX_HT * ADAPTIVE_MAX_WD) >> 6;
  //       s_hevcd_video_decode_ip.u4_8x8_blk_type_map_size =(ADAPTIVE_MAX_HT * ADAPTIVE_MAX_WD) >> 6;

  //       ret = ivd_cxa_api_function((iv_obj_t *)decoder->codec_obj, (void *)&s_hevcd_video_decode_ip, (void *)&s_hevcd_video_decode_op);

  //       if(ret != IV_SUCCESS)
  //       {
  //         err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "video Frame decode FAIL"};
  //         assert(0);
  //       }

  //       decoder->total_bytes_comsumed += ps_video_decode_op->u4_num_bytes_consumed;
  //       decoder->u4_ip_frm_ts++;

  //       if(1 == ps_video_decode_op->u4_output_present)
  //       {
  //         decoder->u4_op_frm_ts++;
  //       }
  //       else
  //       {
  //         if((ps_video_decode_op->u4_error_code >> IVD_FATALERROR) & 1)
  //         {
  //           printf("Fatal error\n");
  //           break;
  //         }
  //       }
  //       // {
  //       //   FILE *fp = NULL;
  //       //   fp = fopen("recon.yuv", "wb");
  //       //   iv_yuv_buf_t s_dump_disp_frm_buf;
  //       //   s_dump_disp_frm_buf = ps_video_decode_op->s_disp_frm_buf;
                      
  //       //   UWORD8 *buf;

  //       //   buf = (UWORD8 *)s_dump_disp_frm_buf.pv_y_buf;
  //       //   for(int i = 0; i < s_dump_disp_frm_buf.u4_y_ht; i++)
  //       //   {
  //       //     fwrite(buf, 1, s_dump_disp_frm_buf.u4_y_wd, fp);
  //       //     buf += s_dump_disp_frm_buf.u4_y_strd;
  //       //   }

  //       //   buf = (UWORD8 *)s_dump_disp_frm_buf.pv_u_buf;
  //       //   for(int i = 0; i < s_dump_disp_frm_buf.u4_u_ht; i++)
  //       //   {
  //       //     fwrite(buf, 1, s_dump_disp_frm_buf.u4_u_wd, fp);
  //       //     buf += s_dump_disp_frm_buf.u4_u_strd;
  //       //   }
  //       //   buf = (UWORD8 *)s_dump_disp_frm_buf.pv_v_buf;
  //       //   for(int i = 0; i < s_dump_disp_frm_buf.u4_v_ht; i++)
  //       //   {
  //       //     fwrite(buf, 1, s_dump_disp_frm_buf.u4_v_wd, fp);
  //       //     buf += s_dump_disp_frm_buf.u4_v_strd;
  //       //   }
  //       //   fclose(fp);
  //       // }
  //     }
  //   }
  // }


  //err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, kSuccess};
  return err;
}

static heif_chroma libhevc_chroma_fmt_to_libheif(IV_COLOR_FORMAT_T output_chroma_format)
{
  switch (output_chroma_format) {
    case IV_GRAY:
      return heif_chroma_monochrome;
    case IV_YUV_420P:
      return heif_chroma_420;
    case IV_YUV_422P:
      return heif_chroma_422;
    case IV_YUV_444P:
      return heif_chroma_444;
    default:
      return heif_chroma_undefined;
    }
}


static struct heif_error convert_libhevc_image_to_heif_image(struct libhevc_decoder* decoder, ivd_video_decode_op_t *ps_video_decode_op, 
                                                             IV_COLOR_FORMAT_T output_chroma_format, struct heif_image** out_img)
{
  bool is_mono = (output_chroma_format == IV_GRAY);
  std::shared_ptr<HeifPixelImage> yuv_img = std::make_shared<HeifPixelImage>();
  heif_chroma chroma_format = libhevc_chroma_fmt_to_libheif(output_chroma_format);
  yuv_img->create(ps_video_decode_op->u4_pic_wd, ps_video_decode_op->u4_pic_ht, 
                  is_mono ? heif_colorspace_monochrome : heif_colorspace_YCbCr, chroma_format);
  // --- transfer data from de265_image to HeifPixelImage
  heif_channel channel2plane[3] = {heif_channel_Y, heif_channel_Cb, heif_channel_Cr};
  int bpp = 8; //libhevc only support 8 bit depth;
  int nPlanes = (is_mono ? 1 : 3);
  for (int c = 0; c < nPlanes; c++) 
  {
    iv_yuv_buf_t s_dump_disp_frm_buf;
    s_dump_disp_frm_buf = ps_video_decode_op->s_disp_frm_buf;
    const uint8_t* data = (c==0) ? (UWORD8 *)s_dump_disp_frm_buf.pv_y_buf : 
                          ((c==1) ? (UWORD8 *)s_dump_disp_frm_buf.pv_u_buf : (UWORD8 *)s_dump_disp_frm_buf.pv_v_buf);
    int w = (c==0) ? s_dump_disp_frm_buf.u4_y_wd : 
            ((c==1) ? s_dump_disp_frm_buf.u4_u_wd : s_dump_disp_frm_buf.u4_v_wd);
    int h = (c==0) ? s_dump_disp_frm_buf.u4_y_ht : 
            ((c==1) ? s_dump_disp_frm_buf.u4_u_ht : s_dump_disp_frm_buf.u4_v_ht);
    int stride = (c==0) ? s_dump_disp_frm_buf.u4_y_strd : 
                 ((c==1) ? s_dump_disp_frm_buf.u4_u_strd : s_dump_disp_frm_buf.u4_v_strd);
    if (w <= 0 || h <= 0) {
      struct heif_error err = {heif_error_Decoder_plugin_error,  heif_suberror_Invalid_image_size, kEmptyString};
      return err;
    }
    if (!yuv_img->add_plane(channel2plane[c], w, h, bpp)) {
      struct heif_error err = {heif_error_Memory_allocation_error, heif_suberror_Unspecified, "Cannot allocate memory for image plane"};
      return err;
    }
    int dst_stride;
    uint8_t* dst_mem = yuv_img->get_plane(channel2plane[c], &dst_stride);
    int bytes_per_pixel = (bpp + 7) / 8;
    for (int y = 0; y < h; y++) {
      memcpy(dst_mem + y * dst_stride, data + y * stride, w * bytes_per_pixel);
    }
  }
  *out_img = new heif_image;
  (*out_img)->image = yuv_img;
  struct heif_error err = {heif_error_Ok, heif_suberror_Unspecified, kSuccess};
  return err;
}

static struct heif_error libhevc_decode_image(void* decoder_raw, struct heif_image** out_img)
{
  WORD32 ret;
  *out_img = nullptr;
  struct libhevc_decoder* decoder = (struct libhevc_decoder*) decoder_raw;
  struct heif_error err = {heif_error_Ok, heif_suberror_Unspecified, kSuccess};
  while(decoder->u4_op_frm_ts < (s_app_ctx.u4_max_frm_ts + s_app_ctx.disp_delay))
  {
    if(decoder->u4_ip_frm_ts < s_app_ctx.num_disp_buf)
    {
      release_disp_frame(decoder->codec_obj, decoder->u4_ip_frm_ts);
    }
    /***********************************************************************/
    /*   Seek the file to start of current frame, this is equavelent of    */
    /*   having a parcer which tells the start of current frame            */
    /***********************************************************************/
    UWORD8 *pic_cur_ptr = NULL;
    {
      pic_cur_ptr = decoder->pu2_bs_buf + decoder->total_bytes_comsumed;
      if(decoder->u4_bytes_remaining == 0)
      {
        err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "Su4_bytes_remaining 0"};
        assert(0);
      }
      {
        ihevcd_cxa_video_decode_ip_t s_hevcd_video_decode_ip = {};
        ihevcd_cxa_video_decode_op_t s_hevcd_video_decode_op = {};
        ivd_video_decode_ip_t *ps_video_decode_ip = &s_hevcd_video_decode_ip.s_ivd_video_decode_ip_t;
        ivd_video_decode_op_t *ps_video_decode_op = &s_hevcd_video_decode_op.s_ivd_video_decode_op_t;
        ps_video_decode_ip->e_cmd = IVD_CMD_VIDEO_DECODE;
        ps_video_decode_ip->u4_ts = decoder->u4_ip_frm_ts;
        ps_video_decode_ip->pv_stream_buffer = pic_cur_ptr;
        ps_video_decode_ip->u4_num_Bytes = decoder->u4_bytes_remaining;
        ps_video_decode_ip->u4_size = sizeof(ihevcd_cxa_video_decode_ip_t);
        ps_video_decode_ip->s_out_buffer.u4_min_out_buf_size[0] = decoder->ps_out_buf->u4_min_out_buf_size[0];
        ps_video_decode_ip->s_out_buffer.u4_min_out_buf_size[1] = decoder->ps_out_buf->u4_min_out_buf_size[1];
        ps_video_decode_ip->s_out_buffer.u4_min_out_buf_size[2] = decoder->ps_out_buf->u4_min_out_buf_size[2];

        ps_video_decode_ip->s_out_buffer.pu1_bufs[0] = decoder->ps_out_buf->pu1_bufs[0];
        ps_video_decode_ip->s_out_buffer.pu1_bufs[1] = decoder->ps_out_buf->pu1_bufs[1];
        ps_video_decode_ip->s_out_buffer.pu1_bufs[2] = decoder->ps_out_buf->pu1_bufs[2];
        ps_video_decode_ip->s_out_buffer.u4_num_bufs = decoder->ps_out_buf->u4_num_bufs;

        ps_video_decode_op->u4_size = sizeof(ihevcd_cxa_video_decode_op_t);
        s_hevcd_video_decode_ip.pu1_8x8_blk_qp_map = NULL; //pu1_qp_map_buf;
        s_hevcd_video_decode_ip.pu1_8x8_blk_type_map = NULL; //pu1_blk_type_map_buf;
        s_hevcd_video_decode_ip.u4_8x8_blk_qp_map_size = (ADAPTIVE_MAX_HT * ADAPTIVE_MAX_WD) >> 6;
        s_hevcd_video_decode_ip.u4_8x8_blk_type_map_size =(ADAPTIVE_MAX_HT * ADAPTIVE_MAX_WD) >> 6;

        ret = ivd_cxa_api_function((iv_obj_t *)decoder->codec_obj, (void *)&s_hevcd_video_decode_ip, (void *)&s_hevcd_video_decode_op);

        if(ret != IV_SUCCESS)
        {
          err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "video Frame decode FAIL"};
          assert(0);
        }

        decoder->total_bytes_comsumed += ps_video_decode_op->u4_num_bytes_consumed;
        decoder->u4_ip_frm_ts++;

        if(1 == ps_video_decode_op->u4_output_present)
        {
          decoder->u4_op_frm_ts++;
          if(*out_img)
          {
            heif_image_release(*out_img);
          }
          err = convert_libhevc_image_to_heif_image(decoder, ps_video_decode_op, s_app_ctx.e_output_chroma_format, out_img);
          if (err.code != heif_error_Ok) {
            return err;
          }
          struct heif_color_profile_nclx* nclx = heif_nclx_color_profile_alloc();
          heif_image_set_nclx_color_profile(*out_img, nclx);
          heif_nclx_color_profile_free(nclx);
        }
        else
        {
          if((ps_video_decode_op->u4_error_code >> IVD_FATALERROR) & 1)
          {
            err = {heif_error_Decoder_plugin_error, heif_suberror_Unspecified, "decode Fatal error"};
            return err;
            //printf("Fatal error\n");
            //break;
          }
        }
        // {
        //   FILE *fp = NULL;
        //   fp = fopen("recon.yuv", "wb");
        //   iv_yuv_buf_t s_dump_disp_frm_buf;
        //   s_dump_disp_frm_buf = ps_video_decode_op->s_disp_frm_buf;
                      
        //   UWORD8 *buf;

        //   buf = (UWORD8 *)s_dump_disp_frm_buf.pv_y_buf;
        //   for(int i = 0; i < s_dump_disp_frm_buf.u4_y_ht; i++)
        //   {
        //     fwrite(buf, 1, s_dump_disp_frm_buf.u4_y_wd, fp);
        //     buf += s_dump_disp_frm_buf.u4_y_strd;
        //   }

        //   buf = (UWORD8 *)s_dump_disp_frm_buf.pv_u_buf;
        //   for(int i = 0; i < s_dump_disp_frm_buf.u4_u_ht; i++)
        //   {
        //     fwrite(buf, 1, s_dump_disp_frm_buf.u4_u_wd, fp);
        //     buf += s_dump_disp_frm_buf.u4_u_strd;
        //   }
        //   buf = (UWORD8 *)s_dump_disp_frm_buf.pv_v_buf;
        //   for(int i = 0; i < s_dump_disp_frm_buf.u4_v_ht; i++)
        //   {
        //     fwrite(buf, 1, s_dump_disp_frm_buf.u4_v_wd, fp);
        //     buf += s_dump_disp_frm_buf.u4_v_strd;
        //   }
        //   fclose(fp);
        // }
      }
    }
  }
  return err;
}

void libhevc_set_strict_decoding(void* decoder_raw, int flag)
{
  struct libhevc_decoder* decoder = (libhevc_decoder*) decoder_raw;

  decoder->strict_decoding = flag;
}

void codec_exit(CHAR *pc_err_message)
{
    printf("%s\n", pc_err_message);
    exit(-1);
}

static void libhevc_free_decoder(void* decoder_raw)
{
  WORD32 ret;

  struct libhevc_decoder* decoder = (struct libhevc_decoder*) decoder_raw;

  ivd_delete_ip_t s_delete_dec_ip;
  ivd_delete_op_t s_delete_dec_op;

  s_delete_dec_ip.e_cmd = IVD_CMD_DELETE;
  s_delete_dec_ip.u4_size = sizeof(ivd_delete_ip_t);
  s_delete_dec_op.u4_size = sizeof(ivd_delete_op_t);

  ret = ivd_cxa_api_function((iv_obj_t *)decoder->codec_obj, (void *)&s_delete_dec_ip,(void *)&s_delete_dec_op);

  if(IV_SUCCESS != ret)
  {
    CHAR ac_error_str[STRLENGTH];
    sprintf(ac_error_str, "Error in Codec delete");
    codec_exit(ac_error_str);
  }

  if(0 == s_app_ctx.share_disp_buf)
  {
    free(decoder->ps_out_buf->pu1_bufs[0]);
  }

  for(int i = 0; i < s_app_ctx.num_disp_buf; i++)
  {
    free(s_app_ctx.s_disp_buffers[i].pu1_bufs[0]);
  }

  free(decoder->ps_out_buf);
  free(decoder->pu1_bs_buf);
  free(decoder->pu2_bs_buf);

}

static const struct heif_decoder_plugin decoder_libhevc
{
    2,
    libhevc_plugin_name,
    libhevc_init_plugin,
    libhevc_deinit_plugin,
    libhevc_does_support_format,
    libhevc_new_decoder,
    libhevc_free_decoder,
    libhevc_push_data,
    libhevc_decode_image,
    libhevc_set_strict_decoding
};


const struct heif_decoder_plugin* get_decoder_plugin_libhevc()
{
  return &decoder_libhevc;
}
