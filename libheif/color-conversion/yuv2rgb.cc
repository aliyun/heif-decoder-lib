/*
 * HEIF codec.
 * Copyright (c) 2023 Dirk Farin <dirk.farin@gmail.com>
 *
 * This file is part of libheif.
 *
 * libheif is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * libheif is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libheif.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cmath>
#include <cstring>
#include "yuv2rgb.h"
#include "nclx.h"
#include "common_utils.h"


template<class Pixel>
std::vector<ColorStateWithCost>
Op_YCbCr_to_RGB<Pixel>::state_after_conversion(const ColorState& input_state,
                                               const ColorState& target_state,
                                               const heif_color_conversion_options& options) const
{
  // this Op only implements the nearest-neighbor algorithm

  if (input_state.chroma != heif_chroma_444) {
    if (options.preferred_chroma_upsampling_algorithm != heif_chroma_upsampling_nearest_neighbor &&
        options.only_use_preferred_chroma_algorithm) {
      return {};
    }
  }

  if (input_state.colorspace != heif_colorspace_YCbCr ||
      (input_state.chroma != heif_chroma_444 &&
       input_state.chroma != heif_chroma_422 &&
       input_state.chroma != heif_chroma_420)) {
    return {};
  }

  int matrix = input_state.nclx_profile.get_matrix_coefficients();
  if ( matrix == 11 || matrix == 14) {
    return {};
  }


  bool hdr = !std::is_same<Pixel, uint8_t>::value;

  if ((input_state.bits_per_pixel != 8) != hdr) {
    return {};
  }

  std::vector<ColorStateWithCost> states;

  ColorState output_state;

  // --- convert to RGB

  output_state.colorspace = heif_colorspace_RGB;
  output_state.chroma = heif_chroma_444;
  output_state.has_alpha = input_state.has_alpha;  // we simply keep the old alpha plane
  output_state.bits_per_pixel = input_state.bits_per_pixel;

  states.push_back({output_state, SpeedCosts_Unoptimized});

  return states;
}


template<class Pixel>
std::shared_ptr<HeifPixelImage>
Op_YCbCr_to_RGB<Pixel>::convert_colorspace(const std::shared_ptr<const HeifPixelImage>& input,
                                           const ColorState& input_state,
                                           const ColorState& target_state,
                                           const heif_color_conversion_options& options) const
{
  bool hdr = !std::is_same<Pixel, uint8_t>::value;

  heif_chroma chroma = input->get_chroma_format();

  int bpp_y = input->get_bits_per_pixel(heif_channel_Y);
  int bpp_cb = input->get_bits_per_pixel(heif_channel_Cb);
  int bpp_cr = input->get_bits_per_pixel(heif_channel_Cr);
  int bpp_a = 0;

  bool has_alpha = input->has_channel(heif_channel_Alpha);

  if (has_alpha) {
    bpp_a = input->get_bits_per_pixel(heif_channel_Alpha);
  }

  if (!hdr) {
    if (bpp_y != 8 ||
        bpp_cb != 8 ||
        bpp_cr != 8) {
      return nullptr;
    }
  }
  else {
    if (bpp_y == 8 ||
        bpp_cb == 8 ||
        bpp_cr == 8) {
      return nullptr;
    }
  }


  if (bpp_y != bpp_cb ||
      bpp_y != bpp_cr) {
    // TODO: test with varying bit depths when we have a test image
    return nullptr;
  }


  auto colorProfile = input->get_color_profile_nclx();

  int width = input->get_width();
  int height = input->get_height();

  auto outimg = std::make_shared<HeifPixelImage>();

  outimg->create(width, height, heif_colorspace_RGB, heif_chroma_444);

  if (!outimg->add_plane(heif_channel_R, width, height, bpp_y) ||
      !outimg->add_plane(heif_channel_G, width, height, bpp_y) ||
      !outimg->add_plane(heif_channel_B, width, height, bpp_y)) {
    return nullptr;
  }

  if (has_alpha) {
    if (!outimg->add_plane(heif_channel_Alpha, width, height, bpp_a)) {
      return nullptr;
    }
  }

  const Pixel* in_y, * in_cb, * in_cr, * in_a;
  int in_y_stride = 0, in_cb_stride = 0, in_cr_stride = 0, in_a_stride = 0;

  Pixel* out_r, * out_g, * out_b, * out_a;
  int out_r_stride = 0, out_g_stride = 0, out_b_stride = 0, out_a_stride = 0;

  in_y = (const Pixel*) input->get_plane(heif_channel_Y, &in_y_stride);
  in_cb = (const Pixel*) input->get_plane(heif_channel_Cb, &in_cb_stride);
  in_cr = (const Pixel*) input->get_plane(heif_channel_Cr, &in_cr_stride);
  out_r = (Pixel*) outimg->get_plane(heif_channel_R, &out_r_stride);
  out_g = (Pixel*) outimg->get_plane(heif_channel_G, &out_g_stride);
  out_b = (Pixel*) outimg->get_plane(heif_channel_B, &out_b_stride);

  if (has_alpha) {
    in_a = (const Pixel*) input->get_plane(heif_channel_Alpha, &in_a_stride);
    out_a = (Pixel*) outimg->get_plane(heif_channel_Alpha, &out_a_stride);
  }
  else {
    in_a = nullptr;
    out_a = nullptr;
  }


  uint16_t halfRange = (uint16_t) (1 << (bpp_y - 1));
  int32_t fullRange = (1 << bpp_y) - 1;
  int limited_range_offset_int = 16 << (bpp_y - 8);
  float limited_range_offset = static_cast<float>(limited_range_offset_int);

  int shiftH = chroma_h_subsampling(chroma) - 1;
  int shiftV = chroma_v_subsampling(chroma) - 1;

  if (hdr) {
    in_y_stride /= 2;
    in_cb_stride /= 2;
    in_cr_stride /= 2;
    in_a_stride /= 2;
    out_r_stride /= 2;
    out_g_stride /= 2;
    out_b_stride /= 2;
    out_a_stride /= 2;
  }

  int matrix_coeffs = 2;
  bool full_range_flag = true;
  YCbCr_to_RGB_coefficients coeffs = YCbCr_to_RGB_coefficients::defaults();
  if (colorProfile) {
    matrix_coeffs = colorProfile->get_matrix_coefficients();
    full_range_flag = colorProfile->get_full_range_flag();
    coeffs = get_YCbCr_to_RGB_coefficients(colorProfile->get_matrix_coefficients(),
                                           colorProfile->get_colour_primaries());
  }


  int x, y;
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      int cx = (x >> shiftH);
      int cy = (y >> shiftV);

      if (matrix_coeffs == 0) {
        if (full_range_flag) {
          out_r[y * out_r_stride + x] = in_cr[cy * in_cr_stride + cx];
          out_g[y * out_g_stride + x] = in_y[y * in_y_stride + x];
          out_b[y * out_b_stride + x] = in_cb[cy * in_cb_stride + cx];
        }
        else {
          // Convert from limited range to full range.
          out_r[y * out_r_stride + x] = (Pixel) clip_f_u16((in_cr[cy * in_cr_stride + cx] - limited_range_offset) * 1.1429f, fullRange);
          out_g[y * out_g_stride + x] = (Pixel) clip_f_u16((in_y[y * in_y_stride + x] - limited_range_offset) * 1.1689f, fullRange);
          out_b[y * out_b_stride + x] = (Pixel) clip_f_u16((in_cb[cy * in_cb_stride + cx] - limited_range_offset) * 1.1429f, fullRange);
        }
      }
      else if (matrix_coeffs == 8) {
        // TODO: check this. I have no input image yet which is known to be correct.
        // TODO: is there a coeff=8 with full_range=false ?

        int yv = in_y[y * in_y_stride + x];
        int cb = in_cb[cy * in_cb_stride + cx] - halfRange;
        int cr = in_cr[cy * in_cr_stride + cx] - halfRange;

        out_r[y * out_r_stride + x] = (Pixel) (clip_int_u8(yv - cb + cr));
        out_g[y * out_g_stride + x] = (Pixel) (clip_int_u8(yv + cb));
        out_b[y * out_b_stride + x] = (Pixel) (clip_int_u8(yv - cb - cr));
      }
      else { // TODO: matrix_coefficients = 11,14
        float yv, cb, cr;
        yv = static_cast<float>(in_y[y * in_y_stride + x] );
        cb = static_cast<float>(in_cb[cy * in_cb_stride + cx] - halfRange);
        cr = static_cast<float>(in_cr[cy * in_cr_stride + cx] - halfRange);

        if (!full_range_flag) {
          yv = (yv - limited_range_offset) * 1.1689f;
          cb = cb * 1.1429f;
          cr = cr * 1.1429f;
        }

        out_r[y * out_r_stride + x] = (Pixel) (clip_f_u16(yv + coeffs.r_cr * cr, fullRange));
        out_g[y * out_g_stride + x] = (Pixel) (clip_f_u16(yv + coeffs.g_cb * cb + coeffs.g_cr * cr, fullRange));
        out_b[y * out_b_stride + x] = (Pixel) (clip_f_u16(yv + coeffs.b_cb * cb, fullRange));
      }
    }

    if (has_alpha) {
      int copyWidth = (hdr ? width * 2 : width);
      memcpy(&out_a[y * out_a_stride], &in_a[y * in_a_stride], copyWidth);
    }
  }

  return outimg;
}

template class Op_YCbCr_to_RGB<uint8_t>;
template class Op_YCbCr_to_RGB<uint16_t>;


std::vector<ColorStateWithCost>
Op_YCbCr420_to_RGB24::state_after_conversion(const ColorState& input_state,
                                             const ColorState& target_state,
                                             const heif_color_conversion_options& options) const
{
  // this Op only implements the nearest-neighbor algorithm

  if (input_state.chroma != heif_chroma_444) {
    if (options.preferred_chroma_upsampling_algorithm != heif_chroma_upsampling_nearest_neighbor &&
        options.only_use_preferred_chroma_algorithm) {
      return {};
    }
  }

  if (input_state.colorspace != heif_colorspace_YCbCr ||
      input_state.chroma != heif_chroma_420 ||
      input_state.bits_per_pixel != 8 ||
      input_state.has_alpha == true) {
    return {};
  }

  int matrix = input_state.nclx_profile.get_matrix_coefficients();
  if (matrix == 0 || matrix == 8 || matrix == 11 || matrix == 14) {
    return {};
  }
  if (!input_state.nclx_profile.get_full_range_flag()) {
    return {};
  }

  std::vector<ColorStateWithCost> states;

  ColorState output_state;

  // --- convert to RGB

  output_state.colorspace = heif_colorspace_RGB;
  output_state.chroma = heif_chroma_interleaved_RGB;
  output_state.has_alpha = false;
  output_state.bits_per_pixel = 8;

  states.push_back({output_state, SpeedCosts_Unoptimized});

  return states;
}


std::shared_ptr<HeifPixelImage>
Op_YCbCr420_to_RGB24::convert_colorspace(const std::shared_ptr<const HeifPixelImage>& input,
                                         const ColorState& input_state,
                                         const ColorState& target_state,
                                         const heif_color_conversion_options& options) const
{
  if (input->get_bits_per_pixel(heif_channel_Y) != 8 ||
      input->get_bits_per_pixel(heif_channel_Cb) != 8 ||
      input->get_bits_per_pixel(heif_channel_Cr) != 8) {
    return nullptr;
  }

  auto outimg = std::make_shared<HeifPixelImage>();

  int width = input->get_width();
  int height = input->get_height();

  outimg->create(width, height, heif_colorspace_RGB, heif_chroma_interleaved_24bit);

  if (!outimg->add_plane(heif_channel_interleaved, width, height, 8)) {
    return nullptr;
  }

  auto colorProfile = input->get_color_profile_nclx();
  YCbCr_to_RGB_coefficients coeffs = YCbCr_to_RGB_coefficients::defaults();
  if (colorProfile) {
    coeffs = get_YCbCr_to_RGB_coefficients(colorProfile->get_matrix_coefficients(),
                                           colorProfile->get_colour_primaries());
  }

  int r_cr = static_cast<int>(std::lround(256 * coeffs.r_cr));
  int g_cr = static_cast<int>(std::lround(256 * coeffs.g_cr));
  int g_cb = static_cast<int>(std::lround(256 * coeffs.g_cb));
  int b_cb = static_cast<int>(std::lround(256 * coeffs.b_cb));

  const uint8_t* in_y, * in_cb, * in_cr;
  int in_y_stride = 0, in_cb_stride = 0, in_cr_stride = 0;

  uint8_t* out_p;
  int out_p_stride = 0;

  in_y = input->get_plane(heif_channel_Y, &in_y_stride);
  in_cb = input->get_plane(heif_channel_Cb, &in_cb_stride);
  in_cr = input->get_plane(heif_channel_Cr, &in_cr_stride);
  out_p = outimg->get_plane(heif_channel_interleaved, &out_p_stride);

  int x, y;
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      int yv = (in_y[y * in_y_stride + x]);
      int cb = (in_cb[y / 2 * in_cb_stride + x / 2] - 128);
      int cr = (in_cr[y / 2 * in_cr_stride + x / 2] - 128);

      out_p[y * out_p_stride + 3 * x + 0] = clip_int_u8(yv + ((r_cr * cr + 128) >> 8));
      out_p[y * out_p_stride + 3 * x + 1] = clip_int_u8(yv + ((g_cb * cb + g_cr * cr + 128) >> 8));
      out_p[y * out_p_stride + 3 * x + 2] = clip_int_u8(yv + ((b_cb * cb + 128) >> 8));
    }
  }

  return outimg;
}


std::vector<ColorStateWithCost>
Op_YCbCr420_to_RGB32::state_after_conversion(const ColorState& input_state,
                                             const ColorState& target_state,
                                             const heif_color_conversion_options& options) const
{
  // this Op only implements the nearest-neighbor algorithm

  if (input_state.chroma != heif_chroma_444) {
    if (options.preferred_chroma_upsampling_algorithm != heif_chroma_upsampling_nearest_neighbor &&
        options.only_use_preferred_chroma_algorithm) {
      return {};
    }
  }

  // Note: no input alpha channel required. It will be filled up with 0xFF.

  if (input_state.colorspace != heif_colorspace_YCbCr ||
      input_state.chroma != heif_chroma_420 ||
      input_state.bits_per_pixel != 8) {
    return {};
  }

  int matrix = input_state.nclx_profile.get_matrix_coefficients();
  if (matrix == 0 || matrix == 8 || matrix == 11 || matrix == 14) {
    return {};
  }
  if (!input_state.nclx_profile.get_full_range_flag()) {
    return {};
  }

  std::vector<ColorStateWithCost> states;

  ColorState output_state;

  // --- convert to RGB

  output_state.colorspace = heif_colorspace_RGB;
  output_state.chroma = heif_chroma_interleaved_RGBA;
  output_state.has_alpha = true;
  output_state.bits_per_pixel = 8;

  states.push_back({output_state, SpeedCosts_Unoptimized});

  return states;
}


std::shared_ptr<HeifPixelImage>
Op_YCbCr420_to_RGB32::convert_colorspace(const std::shared_ptr<const HeifPixelImage>& input,
                                         const ColorState& input_state,
                                         const ColorState& target_state,
                                         const heif_color_conversion_options& options) const
{
  if (input->get_bits_per_pixel(heif_channel_Y) != 8 ||
      input->get_bits_per_pixel(heif_channel_Cb) != 8 ||
      input->get_bits_per_pixel(heif_channel_Cr) != 8) {
    return nullptr;
  }

  auto outimg = std::make_shared<HeifPixelImage>();

  int width = input->get_width();
  int height = input->get_height();

  outimg->create(width, height, heif_colorspace_RGB, heif_chroma_interleaved_32bit);

  if (!outimg->add_plane(heif_channel_interleaved, width, height, 8)) {
    return nullptr;
  }


  // --- get conversion coefficients

  auto colorProfile = input->get_color_profile_nclx();
  YCbCr_to_RGB_coefficients coeffs = YCbCr_to_RGB_coefficients::defaults();
  if (colorProfile) {
    coeffs = get_YCbCr_to_RGB_coefficients(colorProfile->get_matrix_coefficients(),
                                           colorProfile->get_colour_primaries());
  }

  int r_cr = static_cast<int>(std::lround(256 * coeffs.r_cr));
  int g_cr = static_cast<int>(std::lround(256 * coeffs.g_cr));
  int g_cb = static_cast<int>(std::lround(256 * coeffs.g_cb));
  int b_cb = static_cast<int>(std::lround(256 * coeffs.b_cb));


  const bool with_alpha = input->has_channel(heif_channel_Alpha);

  const uint8_t* in_y, * in_cb, * in_cr, * in_a = nullptr;
  int in_y_stride = 0, in_cb_stride = 0, in_cr_stride = 0, in_a_stride = 0;

  uint8_t* out_p;
  int out_p_stride = 0;

  in_y = input->get_plane(heif_channel_Y, &in_y_stride);
  in_cb = input->get_plane(heif_channel_Cb, &in_cb_stride);
  in_cr = input->get_plane(heif_channel_Cr, &in_cr_stride);
  if (with_alpha) {
    in_a = input->get_plane(heif_channel_Alpha, &in_a_stride);
  }

  out_p = outimg->get_plane(heif_channel_interleaved, &out_p_stride);

  int x, y;
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {

      int yv = (in_y[y * in_y_stride + x]);
      int cb = (in_cb[y / 2 * in_cb_stride + x / 2] - 128);
      int cr = (in_cr[y / 2 * in_cr_stride + x / 2] - 128);

      out_p[y * out_p_stride + 4 * x + 0] = clip_int_u8(yv + ((r_cr * cr + 128) >> 8));
      out_p[y * out_p_stride + 4 * x + 1] = clip_int_u8(yv + ((g_cb * cb + g_cr * cr + 128) >> 8));
      out_p[y * out_p_stride + 4 * x + 2] = clip_int_u8(yv + ((b_cb * cb + 128) >> 8));


      if (with_alpha) {
        out_p[y * out_p_stride + 4 * x + 3] = in_a[y * in_a_stride + x];
      }
      else {
        out_p[y * out_p_stride + 4 * x + 3] = 0xFF;
      }
    }
  }

  return outimg;
}


std::vector<ColorStateWithCost>
Op_YCbCr420_to_RRGGBBaa::state_after_conversion(const ColorState& input_state,
                                                const ColorState& target_state,
                                                const heif_color_conversion_options& options) const
{
  // this Op only implements the nearest-neighbor algorithm

  if (input_state.chroma != heif_chroma_444) {
    if (options.preferred_chroma_upsampling_algorithm != heif_chroma_upsampling_nearest_neighbor &&
        options.only_use_preferred_chroma_algorithm) {
      return {};
    }
  }

  if (input_state.colorspace != heif_colorspace_YCbCr ||
      input_state.chroma != heif_chroma_420 ||
      input_state.bits_per_pixel == 8) {
    return {};
  }

  int matrix = input_state.nclx_profile.get_matrix_coefficients();
  if (matrix == 0 || matrix == 8 || matrix == 11 || matrix == 14) {
    return {};
  }

  std::vector<ColorStateWithCost> states;

  ColorState output_state;

  // --- convert to YCbCr

  output_state.colorspace = heif_colorspace_RGB;
  output_state.chroma = (input_state.has_alpha ?
                         heif_chroma_interleaved_RRGGBBAA_LE : heif_chroma_interleaved_RRGGBB_LE);
  output_state.has_alpha = input_state.has_alpha;
  output_state.bits_per_pixel = input_state.bits_per_pixel;

  states.push_back({output_state, SpeedCosts_Unoptimized});


  output_state.colorspace = heif_colorspace_RGB;
  output_state.chroma = (input_state.has_alpha ?
                         heif_chroma_interleaved_RRGGBBAA_BE : heif_chroma_interleaved_RRGGBB_BE);
  output_state.has_alpha = input_state.has_alpha;
  output_state.bits_per_pixel = input_state.bits_per_pixel;

  states.push_back({output_state, SpeedCosts_Unoptimized});

  return states;
}


std::shared_ptr<HeifPixelImage>
Op_YCbCr420_to_RRGGBBaa::convert_colorspace(const std::shared_ptr<const HeifPixelImage>& input,
                                            const ColorState& input_state,
                                            const ColorState& target_state,
                                            const heif_color_conversion_options& options) const
{
  int width = input->get_width();
  int height = input->get_height();

  int bpp = input->get_bits_per_pixel(heif_channel_Y);
  bool has_alpha = input->has_channel(heif_channel_Alpha);

  int le = (target_state.chroma == heif_chroma_interleaved_RRGGBB_LE ||
            target_state.chroma == heif_chroma_interleaved_RRGGBBAA_LE) ? 1 : 0;

  auto outimg = std::make_shared<HeifPixelImage>();
  outimg->create(width, height, heif_colorspace_RGB, target_state.chroma);

  int bytesPerPixel = has_alpha ? 8 : 6;

  if (!outimg->add_plane(heif_channel_interleaved, width, height, bpp)) {
    return nullptr;
  }

  if (has_alpha) {
    if (!outimg->add_plane(heif_channel_Alpha, width, height, bpp)) {
      return nullptr;
    }
  }

  uint8_t* out_p;
  int out_p_stride = 0;

  const uint16_t* in_y, * in_cb, * in_cr, * in_a = nullptr;
  int in_y_stride = 0, in_cb_stride = 0, in_cr_stride = 0, in_a_stride = 0;

  out_p = outimg->get_plane(heif_channel_interleaved, &out_p_stride);
  in_y = (uint16_t*) input->get_plane(heif_channel_Y, &in_y_stride);
  in_cb = (uint16_t*) input->get_plane(heif_channel_Cb, &in_cb_stride);
  in_cr = (uint16_t*) input->get_plane(heif_channel_Cr, &in_cr_stride);

  if (has_alpha) {
    in_a = (uint16_t*) input->get_plane(heif_channel_Alpha, &in_a_stride);
  }

  int maxval = (1 << bpp) - 1;

  bool full_range_flag = true;
  YCbCr_to_RGB_coefficients coeffs = YCbCr_to_RGB_coefficients::defaults();

  auto colorProfile = input->get_color_profile_nclx();
  if (colorProfile) {
    full_range_flag = colorProfile->get_full_range_flag();
    coeffs = get_YCbCr_to_RGB_coefficients(colorProfile->get_matrix_coefficients(),
                                           colorProfile->get_colour_primaries());
  }

  float limited_range_offset = static_cast<float>(16 << (bpp - 8));

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {

      float y_ = in_y[y * in_y_stride / 2 + x];
      float cb = static_cast<float>(in_cb[y / 2 * in_cb_stride / 2 + x / 2] - (1 << (bpp - 1)));
      float cr = static_cast<float>(in_cr[y / 2 * in_cr_stride / 2 + x / 2] - (1 << (bpp - 1)));

      if (!full_range_flag) {
        y_ = (y_ - limited_range_offset) * 1.1689f;
        cb = cb * 1.1429f;
        cr = cr * 1.1429f;
      }

      int r = clip_f_u16(y_ + coeffs.r_cr * cr, maxval);
      int g = clip_f_u16(y_ + coeffs.g_cb * cb + coeffs.g_cr * cr, maxval);
      int b = clip_f_u16(y_ + coeffs.b_cb * cb, maxval);

      out_p[y * out_p_stride + bytesPerPixel * x + 0 + le] = (uint8_t) (r >> 8);
      out_p[y * out_p_stride + bytesPerPixel * x + 2 + le] = (uint8_t) (g >> 8);
      out_p[y * out_p_stride + bytesPerPixel * x + 4 + le] = (uint8_t) (b >> 8);

      out_p[y * out_p_stride + bytesPerPixel * x + 1 - le] = (uint8_t) (r & 0xff);
      out_p[y * out_p_stride + bytesPerPixel * x + 3 - le] = (uint8_t) (g & 0xff);
      out_p[y * out_p_stride + bytesPerPixel * x + 5 - le] = (uint8_t) (b & 0xff);

      if (has_alpha) {
        out_p[y * out_p_stride + 8 * x + 6 + le] = (uint8_t) (in_a[y * in_a_stride / 2 + x] >> 8);
        out_p[y * out_p_stride + 8 * x + 7 - le] = (uint8_t) (in_a[y * in_a_stride / 2 + x] & 0xff);
      }
    }
  }


  return outimg;
}

#if HAVE_YUV
using namespace libyuv;

template<class PIXEL>
int convert_colorspace_core(const bool        has_alpha,
                            const heif_chroma in_chroma,
                            const PIXEL*      in_y, 
                            const int         in_y_stride,
                            const PIXEL*      in_cb,
                            const int         in_cb_stride,
                            const PIXEL*      in_cr,
                            const int         in_cr_stride,
                            const PIXEL*      in_a,
                            const int         in_a_stride,
                            const struct      YuvConstants * matrixYUV,
                            const struct      YuvConstants * matrixYVU,
                            const int         width,
                            const int         height,
                                  uint8_t*    out_p,
                                  int&        out_p_stride
                            )
{
  return 0 ;
}


template< >
int convert_colorspace_core<uint8_t> (const bool        has_alpha,
                                      const heif_chroma in_chroma,
                                      const uint8_t*    in_y, 
                                      const int         in_y_stride,
                                      const uint8_t*    in_cb,
                                      const int         in_cb_stride,
                                      const uint8_t*    in_cr,
                                      const int         in_cr_stride,
                                      const uint8_t*    in_a,
                                      const int         in_a_stride,
                                      const struct      YuvConstants * matrixYUV,
                                      const struct      YuvConstants * matrixYVU,
                                      const int         width,
                                      const int         height,
                                            uint8_t*    out_p,
                                            int&        out_p_stride
                                      )
{
  int hit = 0;

  // AVIF_RGB_FORMAT_RGB   *ToRGB24Matrix  matrixYVU
  if(has_alpha) 
  {
      if (in_chroma == heif_chroma_444 ) {
          if (libyuv::I444AlphaToABGRMatrix(in_y,
                                            in_y_stride,
                                            in_cb,
                                            in_cb_stride,
                                            in_cr,
                                            in_cr_stride,
                                            in_a,
                                            in_a_stride,
                                            out_p,
                                            out_p_stride,
                                            matrixY,
                                            width,
                                            height,
                                            0) == 0) {
              hit = 1;
          }
      }
      else if(in_chroma == heif_chroma_422 ) {
          if (libyuv::I422AlphaToABGRMatrix(in_y,
                                            in_y_stride,
                                            in_cb,
                                            in_cb_stride,
                                            in_cr,
                                            in_cr_stride,
                                            in_a,
                                            in_a_stride,
                                            out_p,
                                            out_p_stride,
                                            matrixY,
                                            width,
                                            height,
                                            0) == 0) {
              hit = 1;
          }
      }
      else if(in_chroma == heif_chroma_420) {
          if (libyuv::I420AlphaToABGRMatrix(in_y,
                                            in_y_stride,
                                            in_cb,
                                            in_cb_stride,
                                            in_cr,
                                            in_cr_stride,
                                            in_a,
                                            in_a_stride,
                                            out_p,
                                            out_p_stride,
                                            matrixY,
                                            width,
                                            height,
                                            0) == 0) {
              hit = 1;
          }
      }
  }
  else 
  {
      if (in_chroma == heif_chroma_444 ) {
          if (libyuv::I444ToARGBMatrix(in_y,
                                       in_y_stride,
                                       in_cr,
                                       in_cr_stride,
                                       in_cb,
                                       in_cb_stride,
                                       out_p,
                                       out_p_stride,
                                       matrixYVU,
                                       width,
                                       height) == 0) {
              hit = 1;
          }
      }
      else if(in_chroma == heif_chroma_422 ) {
          if (libyuv::I422ToARGBMatrix(in_y,
                                       in_y_stride,
                                       in_cr,
                                       in_cr_stride,
                                       in_cb,
                                       in_cb_stride,
                                       out_p,
                                       out_p_stride,
                                       matrixYVU,
                                       width,
                                       height) == 0) {
              hit = 1;
          }
      }
      else if(in_chroma == heif_chroma_420) {
          if (libyuv::I420ToARGBMatrix(in_y,
                                       in_y_stride,
                                       in_cr,
                                       in_cr_stride,
                                       in_cb,
                                       in_cb_stride,
                                       out_p,
                                       out_p_stride,
                                       matrixYVU,
                                       width,
                                       height) == 0) {
              hit = 1;
          }
      }
  }
  return hit;
}


template< >
int convert_colorspace_core<uint16_t> (const bool        has_alpha,
                                       const heif_chroma in_chroma,
                                       const uint16_t*   in_y, 
                                       const int         in_y_stride,
                                       const uint16_t*   in_cb,
                                       const int         in_cb_stride,
                                       const uint16_t*   in_cr,
                                       const int         in_cr_stride,
                                       const uint16_t*   in_a,
                                       const int         in_a_stride,
                                       const struct      YuvConstants * matrixYUV,
                                       const struct      YuvConstants * matrixYVU,
                                       const int         width,
                                       const int         height,
                                             uint8_t*    out_p,
                                             int&        out_p_stride
                                       )
{
  int hit = 0;

  // AVIF_RGB_FORMAT_RGB   *ToRGB24Matrix  matrixYVU
  if(has_alpha) 
  {
      if (in_chroma == heif_chroma_444 ) {
          if (libyuv::I410AlphaToABGRMatrix(in_y,
                                            in_y_stride/2,
                                            in_cb,
                                            in_cb_stride/2,
                                            in_cr,
                                            in_cr_stride/2,
                                            in_a,
                                            in_a_stride/2,
                                            out_p,
                                            out_p_stride,
                                            matrixY,
                                            width,
                                            height,
                                            0) == 0) {
              hit = 1;
          }
      }
      else if(in_chroma == heif_chroma_422 ) {
          if (libyuv::I210AlphaToABGRMatrix(in_y,
                                            in_y_stride/2,
                                            in_cb,
                                            in_cb_stride/2,
                                            in_cr,
                                            in_cr_stride/2,
                                            in_a,
                                            in_a_stride/2,
                                            out_p,
                                            out_p_stride,
                                            matrixY,
                                            width,
                                            height,
                                            0) == 0) {
              hit = 1;
          }
      }
      else if(in_chroma == heif_chroma_420) {
          if (libyuv::I010AlphaToABGRMatrix(in_y,
                                            in_y_stride/2,
                                            in_cb,
                                            in_cb_stride/2,
                                            in_cr,
                                            in_cr_stride/2,
                                            in_a,
                                            in_a_stride/2,
                                            out_p,
                                            out_p_stride,
                                            matrixY,
                                            width,
                                            height,
                                            0) == 0) {
              hit = 1;
          }
      }
  }
  else 
  {
      if (in_chroma == heif_chroma_444 ) {
          if (libyuv::I410ToARGBMatrix(in_y,
                                       in_y_stride/2,
                                       in_cr,
                                       in_cr_stride/2,
                                       in_cb,
                                       in_cb_stride/2,
                                       out_p,
                                       out_p_stride,
                                       matrixYVU,
                                       width,
                                       height) == 0) {
              hit = 1;
          }
      }
      else if(in_chroma == heif_chroma_422 ) {
          if (libyuv::I210ToARGBMatrix(in_y,
                                       in_y_stride/2,
                                       in_cr,
                                       in_cr_stride/2,
                                       in_cb,
                                       in_cb_stride/2,
                                       out_p,
                                       out_p_stride,
                                       matrixYVU,
                                       width,
                                       height) == 0) {
              hit = 1;
          }
      }
      else if(in_chroma == heif_chroma_420) {
          if (libyuv::I010ToARGBMatrix(in_y,
                                       in_y_stride/2,
                                       in_cr,
                                       in_cr_stride/2,
                                       in_cb,
                                       in_cb_stride/2,
                                       out_p,
                                       out_p_stride,
                                       matrixYVU,
                                       width,
                                       height) == 0) {
              hit = 1;
          }
      }
  }
  return hit;
}


template<class Pixel>
std::shared_ptr<HeifPixelImage> convert_colorspace_libyuv(const std::shared_ptr<const HeifPixelImage>& input,
                                                          const ColorState& target_state,
                                                          const heif_color_conversion_options& options,
                                                          const struct YuvConstants * matrixYUV,
                                                          const struct YuvConstants * matrixYVU)
{
//  (void)matrixYVU;

  bool has_alpha = input->has_channel(heif_channel_Alpha);
  int width = input->get_width();
  int height = input->get_height();
  uint8_t bpp = input->get_bits_per_pixel(heif_channel_Y);
  bool hdr = !std::is_same<Pixel, uint8_t>::value;
  if(hdr){
    if(bpp <= 8) {
      return nullptr;
    }
  }

  heif_chroma  in_chroma = input->get_chroma_format();

  auto outimg = std::make_shared<HeifPixelImage>();
  heif_chroma  dest_chroma = heif_chroma_interleaved_RGBA ;

  // notice : this type change target_state.chroma, from RGB to interleaved_RGBA
  outimg->create(width, height, heif_colorspace_RGB, dest_chroma ); // RGBA
  if(input->get_image_use_external_buf() && (input->get_image_external_buf_base() != NULL)) {
    outimg->set_image_external_info(true, input->get_image_external_buf_base(), 
                                          input->get_image_external_buf_len(), 
                                          input->get_image_external_buf_stride());
    outimg->add_shared_rgba_plane(heif_channel_interleaved, width, height, 8);
  }
  else {
    outimg->add_plane(heif_channel_interleaved, width, height, 8);
  }

  const Pixel* in_y, * in_cb, * in_cr, * in_a;
  int in_y_stride = 0, in_cb_stride = 0, in_cr_stride = 0, in_a_stride = 0;

  uint8_t* out_p;
  int out_p_stride = 0;

  in_y = (const Pixel*) input->get_plane(heif_channel_Y, &in_y_stride);
  in_cb = (const Pixel*) input->get_plane(heif_channel_Cb, &in_cb_stride);
  in_cr = (const Pixel*) input->get_plane(heif_channel_Cr, &in_cr_stride);
  out_p = (uint8_t*) outimg->get_plane(heif_channel_interleaved, &out_p_stride);

  if (has_alpha) {
    in_a = (const Pixel*) input->get_plane(heif_channel_Alpha, &in_a_stride);
  }
  else {
    in_a = nullptr;
  }

  int hit = 0;
  if (target_state.colorspace == heif_colorspace_RGB) {

      hit = convert_colorspace_core<Pixel>(has_alpha,
                                           in_chroma,
                                           in_y,
                                           in_y_stride,
                                           in_cb,
                                           in_cb_stride,
                                           in_cr,
                                           in_cr_stride,
                                           in_a,
                                           in_a_stride,
                                           matrixYUV,
                                           matrixYVU,
                                           width,
                                           height,
                                           out_p,
                                           out_p_stride);

  }

  return (hit)? outimg : nullptr ;
}


std::vector<ColorStateWithCost>
Op_YCbCr_to_RGBA_GENERAL::state_after_conversion(const ColorState& input_state,
                                                 const ColorState& target_state,
                                                 const heif_color_conversion_options& options) const
{
  // Note: if YUV library is not existed, return null.
#if !HAVE_YUV
  return {};
#endif

  // Note: no input alpha channel required. It will be filled up with 0xFF.
  if (input_state.colorspace != heif_colorspace_YCbCr ||
      input_state.chroma != heif_chroma_420) { // ||
//      input_state.bits_per_pixel != 8) {
    return {};
  }

//  if (input_state.nclx_profile) {
//    int matrix = input_state.nclx_profile->get_matrix_coefficients();
//    if (matrix == 0 || matrix == 8 || matrix == 11 || matrix == 14) {
//      return {};
//    }
//    if (!input_state.nclx_profile->get_full_range_flag()) {
//      return {};
//    }
//  }

  std::vector<ColorStateWithCost> states;

  ColorState output_state;

  // --- convert to RGB

  output_state.colorspace = heif_colorspace_RGB;
  output_state.chroma = heif_chroma_interleaved_RGBA;
  output_state.has_alpha = true;
  output_state.bits_per_pixel = 8;

  states.push_back({output_state, SpeedCosts_Trivial});

  return states;
}


std::shared_ptr<HeifPixelImage>
Op_YCbCr_to_RGBA_GENERAL::convert_colorspace(const std::shared_ptr<const HeifPixelImage>& input,
                                             const ColorState& input_state,
                                             const ColorState& target_state,
                                             const heif_color_conversion_options& options) const
{

  int bpp_y = input->get_bits_per_pixel(heif_channel_Y);
  int bpp_cb = input->get_bits_per_pixel(heif_channel_Cb);
  int bpp_cr = input->get_bits_per_pixel(heif_channel_Cr);

  if (bpp_y != bpp_cb ||
      bpp_y != bpp_cr) {
    // TODO: test with varying bit depths when we have a test image
    return nullptr;
  }

  // get YuvConstants coeff 
  auto colorProfile = input->get_color_profile_nclx();

  int matrix_coeffs = 2;
  bool full_range_flag = true;
  int colorprimaries = 2;
  if (colorProfile) {
    matrix_coeffs = colorProfile->get_matrix_coefficients();
    full_range_flag = colorProfile->get_full_range_flag();
    colorprimaries = colorProfile->get_colour_primaries();
  }

  // Find the correct libyuv YuvConstants, based on range and CP/MC
  const struct YuvConstants * matrixYUV = &libyuv::kYuvJPEGConstants;
  const struct YuvConstants * matrixYVU = &libyuv::kYvuJPEGConstants;
//    if (image->yuvRange == AVIF_RANGE_FULL) {
  if ( full_range_flag ) {
      switch (matrix_coeffs ) {
          // BT.709 full range YuvConstants were added in libyuv version 1772.
          // See https://chromium-review.googlesource.com/c/libyuv/libyuv/+/2646472.
          case HEIF_MATRIX_COEFFICIENTS_BT709:
              matrixYUV = &libyuv::kYuvF709Constants;
              matrixYVU = &libyuv::kYvuF709Constants;
              break;
          case HEIF_MATRIX_COEFFICIENTS_BT470BG:
          case HEIF_MATRIX_COEFFICIENTS_BT601:
          case HEIF_MATRIX_COEFFICIENTS_UNSPECIFIED:
              matrixYUV = &libyuv::kYuvJPEGConstants;
              matrixYVU = &libyuv::kYvuJPEGConstants;
              break;
          // BT.2020 full range YuvConstants were added in libyuv version 1775.
          // See https://chromium-review.googlesource.com/c/libyuv/libyuv/+/2678859.
          case HEIF_MATRIX_COEFFICIENTS_BT2020_NCL:
              matrixYUV = &libyuv::kYuvV2020Constants;
              matrixYVU = &libyuv::kYvuV2020Constants;
              break;
          case HEIF_MATRIX_COEFFICIENTS_CHROMA_DERIVED_NCL:
              switch (colorprimaries ) {
                  case HEIF_COLOR_PRIMARIES_BT709:
                  case HEIF_COLOR_PRIMARIES_UNSPECIFIED:
                      matrixYUV = &libyuv::kYuvF709Constants;
                      matrixYVU = &libyuv::kYvuF709Constants;
                      break;
                  case HEIF_COLOR_PRIMARIES_BT470BG:
                  case HEIF_COLOR_PRIMARIES_BT601:
                      matrixYUV = &libyuv::kYuvJPEGConstants;
                      matrixYVU = &libyuv::kYvuJPEGConstants;
                      break;
                  case HEIF_COLOR_PRIMARIES_BT2020:
                      matrixYUV = &libyuv::kYuvV2020Constants;
                      matrixYVU = &libyuv::kYvuV2020Constants;
                      break;

                  case HEIF_COLOR_PRIMARIES_UNKNOWN:
                  case HEIF_COLOR_PRIMARIES_BT470M:
                  case HEIF_COLOR_PRIMARIES_SMPTE240:
                  case HEIF_COLOR_PRIMARIES_GENERIC_FILM:
                  case HEIF_COLOR_PRIMARIES_XYZ:
                  case HEIF_COLOR_PRIMARIES_SMPTE431:
                  case HEIF_COLOR_PRIMARIES_SMPTE432:
                  case HEIF_COLOR_PRIMARIES_EBU3213:
                      break;
              }
              break;

          case HEIF_MATRIX_COEFFICIENTS_IDENTITY:
          case HEIF_MATRIX_COEFFICIENTS_FCC:
          case HEIF_MATRIX_COEFFICIENTS_SMPTE240:
          case HEIF_MATRIX_COEFFICIENTS_YCGCO:
          case HEIF_MATRIX_COEFFICIENTS_BT2020_CL:
          case HEIF_MATRIX_COEFFICIENTS_SMPTE2085:
          case HEIF_MATRIX_COEFFICIENTS_CHROMA_DERIVED_CL:
          case HEIF_MATRIX_COEFFICIENTS_ICTCP:
              break;
      }
  } else {
      switch (matrix_coeffs ) {
          case HEIF_MATRIX_COEFFICIENTS_BT709:
              matrixYUV = &libyuv::kYuvH709Constants;
              matrixYVU = &libyuv::kYvuH709Constants;
              break;
          case HEIF_MATRIX_COEFFICIENTS_BT470BG:
          case HEIF_MATRIX_COEFFICIENTS_BT601:
          case HEIF_MATRIX_COEFFICIENTS_UNSPECIFIED:
              matrixYUV = &libyuv::kYuvI601Constants;
              matrixYVU = &libyuv::kYvuI601Constants;
              break;
          case HEIF_MATRIX_COEFFICIENTS_BT2020_NCL:
              matrixYUV = &libyuv::kYuv2020Constants;
              matrixYVU = &libyuv::kYvu2020Constants;
              break;
          case HEIF_MATRIX_COEFFICIENTS_CHROMA_DERIVED_NCL:
              switch (colorprimaries ) {
                  case HEIF_COLOR_PRIMARIES_BT709:
                  case HEIF_COLOR_PRIMARIES_UNSPECIFIED:
                      matrixYUV = &libyuv::kYuvH709Constants;
                      matrixYVU = &libyuv::kYvuH709Constants;
                      break;
                  case HEIF_COLOR_PRIMARIES_BT470BG:
                  case HEIF_COLOR_PRIMARIES_BT601:
                      matrixYUV = &libyuv::kYuvI601Constants;
                      matrixYVU = &libyuv::kYvuI601Constants;
                      break;
                  case HEIF_COLOR_PRIMARIES_BT2020:
                      matrixYUV = &libyuv::kYuv2020Constants;
                      matrixYVU = &libyuv::kYvu2020Constants;
                      break;

                  case HEIF_COLOR_PRIMARIES_UNKNOWN:
                  case HEIF_COLOR_PRIMARIES_BT470M:
                  case HEIF_COLOR_PRIMARIES_SMPTE240:
                  case HEIF_COLOR_PRIMARIES_GENERIC_FILM:
                  case HEIF_COLOR_PRIMARIES_XYZ:
                  case HEIF_COLOR_PRIMARIES_SMPTE431:
                  case HEIF_COLOR_PRIMARIES_SMPTE432:
                  case HEIF_COLOR_PRIMARIES_EBU3213:
                      break;
              }
              break;
          case HEIF_MATRIX_COEFFICIENTS_IDENTITY:
          case HEIF_MATRIX_COEFFICIENTS_FCC:
          case HEIF_MATRIX_COEFFICIENTS_SMPTE240:
          case HEIF_MATRIX_COEFFICIENTS_YCGCO:
          case HEIF_MATRIX_COEFFICIENTS_BT2020_CL:
          case HEIF_MATRIX_COEFFICIENTS_SMPTE2085:
          case HEIF_MATRIX_COEFFICIENTS_CHROMA_DERIVED_CL:
          case HEIF_MATRIX_COEFFICIENTS_ICTCP:
              break;
      }
  }

  if(!matrixYUV) {
    return nullptr;
  }

  std::shared_ptr<HeifPixelImage> out_img ;
  if(bpp_y <= 8 ) // RGBA
  {
    out_img = convert_colorspace_libyuv<uint8_t>(input, target_state, options, matrixYUV, matrixYVU);
  } 
  else  // RRGGBBAA
  {
    out_img = convert_colorspace_libyuv<uint16_t>(input, target_state, options, matrixYUV, matrixYVU);
  }

  return out_img;
}

#endif