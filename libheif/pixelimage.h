/*
 * HEIF codec.
 * Copyright (c) 2017 Dirk Farin <dirk.farin@gmail.com>
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


#ifndef LIBHEIF_IMAGE_H
#define LIBHEIF_IMAGE_H

//#include "heif.h"
#include "error.h"
#include "nclx.h"

#include <vector>
#include <memory>
#include <map>
#include <set>
#include <utility>


heif_chroma chroma_from_subsampling(int h, int v);

bool is_chroma_with_alpha(heif_chroma chroma);

int num_interleaved_pixels_per_plane(heif_chroma chroma);

bool is_integer_multiple_of_chroma_size(int width,
                                        int height,
                                        heif_chroma chroma);

// Returns the list of valid heif_chroma values for a given colorspace.
std::vector<heif_chroma> get_valid_chroma_values_for_colorspace(heif_colorspace colorspace);

class HeifPixelImage : public std::enable_shared_from_this<HeifPixelImage>,
                       public ErrorBuffer
{
public:
  explicit HeifPixelImage() = default;

  ~HeifPixelImage();

  void create(int width, int height, heif_colorspace colorspace, heif_chroma chroma);

  bool add_plane(heif_channel channel, int width, int height, int bit_depth);

    bool add_shared_rgba_plane(heif_channel channel, int width, int height, uint8_t bit_depth);

  bool has_channel(heif_channel channel) const;

  // Has alpha information either as a separate channel or in the interleaved format.
  bool has_alpha() const;

  bool is_premultiplied_alpha() const { return m_premultiplied_alpha; }

  void set_premultiplied_alpha(bool flag) { m_premultiplied_alpha = flag; }

  int get_width() const { return m_width; }

  int get_height() const { return m_height; }

  int get_width(enum heif_channel channel) const;

  int get_height(enum heif_channel channel) const;

    bool get_image_use_external_buf() const { return image_use_external_buf; }

    uint8_t * get_image_external_buf_base() const { return image_external_buf_base; }

    uint32_t  get_image_external_buf_len() const { return  image_external_buf_len; }

    uint32_t  get_image_external_buf_stride() const { return  image_external_buf_stride; }

  heif_chroma get_chroma_format() const { return m_chroma; }

  heif_colorspace get_colorspace() const { return m_colorspace; }

  std::set<enum heif_channel> get_channel_set() const;

  uint8_t get_storage_bits_per_pixel(enum heif_channel channel) const;

  uint8_t get_bits_per_pixel(enum heif_channel channel) const;

  uint8_t* get_plane(enum heif_channel channel, int* out_stride);

  const uint8_t* get_plane(enum heif_channel channel, int* out_stride) const;

  void copy_new_plane_from(const std::shared_ptr<const HeifPixelImage>& src_image,
                           heif_channel src_channel,
                           heif_channel dst_channel);

  void fill_new_plane(heif_channel dst_channel, uint16_t value, int width, int height, int bpp);

  void transfer_plane_from_image_as(const std::shared_ptr<HeifPixelImage>& source,
                                    heif_channel src_channel,
                                    heif_channel dst_channel);

  Error rotate_ccw(int angle_degrees,
                   std::shared_ptr<HeifPixelImage>& out_img);

  Error mirror_inplace(heif_transform_mirror_direction);

  Error crop(int left, int right, int top, int bottom,
             std::shared_ptr<HeifPixelImage>& out_img) const;

    Error rgba_premultiply_alpha( void ) ;

  Error fill_RGB_16bit(uint16_t r, uint16_t g, uint16_t b, uint16_t a);

  Error overlay(std::shared_ptr<HeifPixelImage>& overlay, int32_t dx, int32_t dy);

  Error scale_nearest_neighbor(std::shared_ptr<HeifPixelImage>& output, int width, int height) const;

    void set_image_resolution(int width, int height) {  m_width = width;  m_height = height; }

  void set_color_profile_nclx(const std::shared_ptr<const color_profile_nclx>& profile) { m_color_profile_nclx = profile; }

  const std::shared_ptr<const color_profile_nclx>& get_color_profile_nclx() const { return m_color_profile_nclx; }

  void set_color_profile_icc(const std::shared_ptr<const color_profile_raw>& profile) { m_color_profile_icc = profile; }
    
    void set_image_external_info(bool use_buf, void* buf_base, uint32_t buf_len, uint32_t buf_stride) { 
      image_use_external_buf = use_buf ;
      image_external_buf_base= (uint8_t *)buf_base ;
      image_external_buf_len = buf_len ;
      image_external_buf_stride = buf_stride ;
    }

  const std::shared_ptr<const color_profile_raw>& get_color_profile_icc() const { return m_color_profile_icc; }

  void debug_dump() const;

  bool extend_padding_to_size(int width, int height);

  // --- pixel aspect ratio

  bool has_nonsquare_pixel_ratio() const { return m_PixelAspectRatio_h != m_PixelAspectRatio_v; }

  void get_pixel_ratio(uint32_t* h, uint32_t* v) const
  {
    *h = m_PixelAspectRatio_h;
    *v = m_PixelAspectRatio_v;
  }

  void set_pixel_ratio(uint32_t h, uint32_t v)
  {
    m_PixelAspectRatio_h = h;
    m_PixelAspectRatio_v = v;
  }

  // --- clli

  bool has_clli() const { return m_clli.max_content_light_level != 0 || m_clli.max_pic_average_light_level != 0; }

  heif_content_light_level get_clli() const { return m_clli; }

  void set_clli(const heif_content_light_level& clli) { m_clli = clli; }

  // --- mdcv

  bool has_mdcv() const { return m_mdcv_set; }

  heif_mastering_display_colour_volume get_mdcv() const { return m_mdcv; }

  void set_mdcv(const heif_mastering_display_colour_volume& mdcv)
  {
    m_mdcv = mdcv;
    m_mdcv_set = true;
  }

  void unset_mdcv() { m_mdcv_set = false; }

  // --- warnings

  void add_warning(Error warning) { m_warnings.emplace_back(std::move(warning)); }

  const std::vector<Error>& get_warnings() const { return m_warnings; }

    void set_duration_in_Timescales(uint64_t duration) {durationInTimescales = duration;}

    uint64_t get_duration_in_Timescales() const {return durationInTimescales;}

private:
  struct ImagePlane
  {
    bool alloc(int width, int height, int bit_depth, heif_chroma chroma);

    uint8_t m_bit_depth = 0;

    // the "visible" area of the plane
    int m_width = 0;
    int m_height = 0;

    // the allocated memory size
    int m_mem_width = 0;
    int m_mem_height = 0;

    uint8_t* mem = nullptr; // aligned memory start
    uint8_t* allocated_mem = nullptr; // unaligned memory we allocated
    uint32_t stride = 0; // bytes per line
      bool plane_use_external_buf = false; 
  };

    uint64_t durationInTimescales = 0;
  int m_width = 0;
  int m_height = 0;
  heif_colorspace m_colorspace = heif_colorspace_undefined;
  heif_chroma m_chroma = heif_chroma_undefined;
  bool m_premultiplied_alpha = false;
  std::shared_ptr<const color_profile_nclx> m_color_profile_nclx;
  std::shared_ptr<const color_profile_raw> m_color_profile_icc;

  std::map<heif_channel, ImagePlane> m_planes;

  uint32_t m_PixelAspectRatio_h = 1;
  uint32_t m_PixelAspectRatio_v = 1;
  heif_content_light_level m_clli{};
  heif_mastering_display_colour_volume m_mdcv{};
  bool m_mdcv_set = false; // replace with std::optional<> when we are on C*+17

  std::vector<Error> m_warnings;

    bool image_use_external_buf = false ; // if image_use_external_buf =true, need to free, if image_use_external_buf =false, 
    uint8_t* image_external_buf_base = NULL;// if image_use_external_buf = true, the buf pointed by ext_image_plane keep pixel data;
    uint32_t image_external_buf_len  = 0 ;  // if image_use_external_buf = true, the length is the buf memory length
    uint32_t image_external_buf_stride=0 ;  // if image_use_external_buf = true, the stride is the buf memory stride
  };

#endif
