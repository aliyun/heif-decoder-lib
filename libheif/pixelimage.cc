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


#include "pixelimage.h"
#include "common_utils.h"
#if HAVE_YUV
#include "libyuv.h"
#endif

#include <cassert>
#include <cstring>
#include <utility>
#include <limits>


heif_chroma chroma_from_subsampling(int h, int v)
{
  if (h == 2 && v == 2) {
    return heif_chroma_420;
  }
  else if (h == 2 && v == 1) {
    return heif_chroma_422;
  }
  else if (h == 1 && v == 1) {
    return heif_chroma_444;
  }
  else {
    assert(false);
    return heif_chroma_undefined;
  }
}


HeifPixelImage::~HeifPixelImage()
{
  for (auto& iter : m_planes) {
    if(!iter.second.plane_use_external_buf) {
      delete[] iter.second.allocated_mem;
    }
  }
}


int num_interleaved_pixels_per_plane(heif_chroma chroma)
{
  switch (chroma) {
    case heif_chroma_undefined:
    case heif_chroma_monochrome:
    case heif_chroma_420:
    case heif_chroma_422:
    case heif_chroma_444:
      return 1;

    case heif_chroma_interleaved_RGB:
    case heif_chroma_interleaved_RRGGBB_BE:
    case heif_chroma_interleaved_RRGGBB_LE:
      return 3;

    case heif_chroma_interleaved_RGBA:
    case heif_chroma_interleaved_RRGGBBAA_BE:
    case heif_chroma_interleaved_RRGGBBAA_LE:
      return 4;
  }

  assert(false);
  return 0;
}


bool is_integer_multiple_of_chroma_size(int width,
                                        int height,
                                        heif_chroma chroma)
{
  switch (chroma) {
    case heif_chroma_444:
    case heif_chroma_monochrome:
      return true;
    case heif_chroma_422:
      return (width & 1) == 0;
    case heif_chroma_420:
      return (width & 1) == 0 && (height & 1) == 0;
    default:
      assert(false);
      return false;
  }
}


std::vector<heif_chroma> get_valid_chroma_values_for_colorspace(heif_colorspace colorspace)
{
  switch (colorspace) {
    case heif_colorspace_YCbCr:
      return {heif_chroma_420, heif_chroma_422, heif_chroma_444};

    case heif_colorspace_RGB:
      return {heif_chroma_444,
              heif_chroma_interleaved_RGB,
              heif_chroma_interleaved_RGBA,
              heif_chroma_interleaved_RRGGBB_BE,
              heif_chroma_interleaved_RRGGBBAA_BE,
              heif_chroma_interleaved_RRGGBB_LE,
              heif_chroma_interleaved_RRGGBBAA_LE};

    case heif_colorspace_monochrome:
      return {heif_chroma_monochrome};

    default:
      return {};
  }
}


void HeifPixelImage::create(int width, int height, heif_colorspace colorspace, heif_chroma chroma)
{
  m_width = width;
  m_height = height;
  m_colorspace = colorspace;
  m_chroma = chroma;
}

static uint32_t rounded_size(uint32_t s)
{
  s = (s + 1U) & ~1U;

  if (s < 64) {
    s = 64;
  }

  return s;
}

bool HeifPixelImage::add_plane(heif_channel channel, int width, int height, int bit_depth)
{
  ImagePlane plane;
  if (plane.alloc(width, height, bit_depth, m_chroma)) {
    m_planes.insert(std::make_pair(channel, plane));
    return true;
  }
  else {
    return false;
  }
}


bool HeifPixelImage::ImagePlane::alloc(int width, int height, int bit_depth, heif_chroma chroma)
{
  assert(width >= 0);
  assert(height >= 0);
  assert(bit_depth >= 1);
  assert(bit_depth <= 32);

  plane_use_external_buf = false; 

  // use 16 byte alignment
  uint16_t alignment = 16; // must be power of two

  m_width = width;
  m_height = height;

  m_mem_width = rounded_size(width);
  m_mem_height = rounded_size(height);

  // for backwards compatibility, allow for 24/32 bits for RGB/RGBA interleaved chromas

  if (chroma == heif_chroma_interleaved_RGB && bit_depth == 24) {
    bit_depth = 8;
  }

  if (chroma == heif_chroma_interleaved_RGBA && bit_depth == 32) {
    bit_depth = 8;
  }

  assert(m_bit_depth <= 16);
  m_bit_depth = static_cast<uint8_t>(bit_depth);


  int bytes_per_component = (m_bit_depth + 7) / 8;
  int bytes_per_pixel = num_interleaved_pixels_per_plane(chroma) * bytes_per_component;

  stride = m_mem_width * bytes_per_pixel;
  stride = (stride + alignment - 1U) & ~(alignment - 1U);

  try {
    allocated_mem = new uint8_t[m_mem_height * stride + alignment - 1];
    mem = allocated_mem;

    // shift beginning of image data to aligned memory position

    auto mem_start_addr = (uint64_t) mem;
    auto mem_start_offset = (mem_start_addr & (alignment - 1U));
    if (mem_start_offset != 0) {
      mem += alignment - mem_start_offset;
    }

    return true;
  }
  catch (const std::bad_alloc& excpt) {
    return false;
  }
}


bool HeifPixelImage::add_shared_rgba_plane(heif_channel channel, int width, int height, uint8_t bit_depth)
{
  ImagePlane plane;
  
  int bytes_per_component = (bit_depth + 7) / 8;
  uint32_t allpixel = width * height * bytes_per_component  ;
  switch(m_chroma ) {
    case  heif_chroma_monochrome :  allpixel = width * height * bytes_per_component ;  break ;
    case  heif_chroma_420 :         allpixel = width * height * bytes_per_component * 1.5;  break ;
    case  heif_chroma_422 :         allpixel = width * height * bytes_per_component * 2 ;  break ;
    case  heif_chroma_undefined :
    case  heif_chroma_444 :         allpixel = width * height * bytes_per_component * 3 ;  break ;
    case  heif_chroma_interleaved_RGB :         allpixel = width * height * bytes_per_component * 3 ;  break ;
    case  heif_chroma_interleaved_RGBA :        allpixel = width * height * bytes_per_component * 4 ;  break ;
    case  heif_chroma_interleaved_RRGGBB_BE :   allpixel = width * height * bytes_per_component * 6 ;  break ;
    case  heif_chroma_interleaved_RRGGBBAA_BE : allpixel = width * height * bytes_per_component * 8 ;  break ;
    case  heif_chroma_interleaved_RRGGBB_LE :   allpixel = width * height * bytes_per_component * 6 ;  break ;
    case  heif_chroma_interleaved_RRGGBBAA_LE : allpixel = width * height * bytes_per_component * 8 ;  break ;
  }

  if( image_external_buf_base && (allpixel <= image_external_buf_len)) {
    plane.m_width = width ;
    plane.m_height= height;
    plane.m_mem_width = width ;
    plane.m_mem_height= height;
    plane.m_bit_depth = bit_depth;

    plane.stride = image_external_buf_stride ;
    plane.mem = image_external_buf_base ;
    plane.allocated_mem = image_external_buf_base ;
    plane.plane_use_external_buf = true ;
    m_planes.insert(std::make_pair(channel, plane));

    return true;
  } 
  else {
    if (plane.alloc(width, height, bit_depth, m_chroma)) {
      m_planes.insert(std::make_pair(channel , plane));
      return true;
    }
    else {
      return false;
    }
  }

}


bool HeifPixelImage::extend_padding_to_size(int width, int height)
{
  for (auto& planeIter : m_planes) {
    auto* plane = &planeIter.second;

    int subsampled_width, subsampled_height;
    get_subsampled_size(width, height, planeIter.first, m_chroma,
                        &subsampled_width, &subsampled_height);

    int old_width = plane->m_width;
    int old_height = plane->m_height;

    if (plane->m_mem_width < subsampled_width ||
        plane->m_mem_height < subsampled_height) {

      ImagePlane newPlane;
      if (!newPlane.alloc(subsampled_width, subsampled_height, plane->m_bit_depth, m_chroma)) {
        return false;
      }

      // copy the visible part of the old plane into the new plane

      for (int y = 0; y < plane->m_height; y++) {
        memcpy(&newPlane.mem[y * newPlane.stride],
               &plane->mem[y * plane->stride],
               plane->m_width);
      }

      planeIter.second = newPlane;
      plane = &planeIter.second;
    }

    // extend plane size

    int bytes_per_pixel = (plane->m_bit_depth + 7) / 8;

    for (int y = 0; y < old_height; y++) {
      for (int x = old_width; x < subsampled_width; x++) {
        memcpy(&plane->mem[y * plane->stride + x * bytes_per_pixel],
               &plane->mem[y * plane->stride + (plane->m_width - 1) * bytes_per_pixel],
               bytes_per_pixel);
      }
    }

    for (int y = old_height; y < subsampled_height; y++) {
      memcpy(&plane->mem[y * plane->stride],
             &plane->mem[(plane->m_height - 1) * plane->stride],
             subsampled_width * bytes_per_pixel);
    }
  }

  // don't modify the logical image size

  return true;
}


bool HeifPixelImage::has_channel(heif_channel channel) const
{
  return (m_planes.find(channel) != m_planes.end());
}


bool HeifPixelImage::has_alpha() const
{
  return has_channel(heif_channel_Alpha) ||
         get_chroma_format() == heif_chroma_interleaved_RGBA ||
         get_chroma_format() == heif_chroma_interleaved_RRGGBBAA_BE ||
         get_chroma_format() == heif_chroma_interleaved_RRGGBBAA_LE;
}


int HeifPixelImage::get_width(enum heif_channel channel) const
{
  auto iter = m_planes.find(channel);
  if (iter == m_planes.end()) {
    return -1;
  }

  return iter->second.m_width;
}


int HeifPixelImage::get_height(enum heif_channel channel) const
{
  auto iter = m_planes.find(channel);
  if (iter == m_planes.end()) {
    return -1;
  }

  return iter->second.m_height;
}


std::set<heif_channel> HeifPixelImage::get_channel_set() const
{
  std::set<heif_channel> channels;

  for (const auto& plane : m_planes) {
    channels.insert(plane.first);
  }

  return channels;
}


uint8_t HeifPixelImage::get_storage_bits_per_pixel(enum heif_channel channel) const
{
  if (channel == heif_channel_interleaved) {
    auto chroma = get_chroma_format();
    switch (chroma) {
      case heif_chroma_interleaved_RGB:
        return 24;
      case heif_chroma_interleaved_RGBA:
        return 32;
      case heif_chroma_interleaved_RRGGBB_BE:
      case heif_chroma_interleaved_RRGGBB_LE:
        return 48;
      case heif_chroma_interleaved_RRGGBBAA_BE:
      case heif_chroma_interleaved_RRGGBBAA_LE:
        return 64;
      default:
        return -1; // invalid channel/chroma specification
    }
  }
  else {
    uint32_t bpp = (get_bits_per_pixel(channel) + 7U) & ~7U;
    assert(bpp <= 255);
    return static_cast<uint8_t>(bpp);
  }
}


uint8_t HeifPixelImage::get_bits_per_pixel(enum heif_channel channel) const
{
  auto iter = m_planes.find(channel);
  if (iter == m_planes.end()) {
    return -1;
  }

  return iter->second.m_bit_depth;
}


uint8_t* HeifPixelImage::get_plane(enum heif_channel channel, int* out_stride)
{
  auto iter = m_planes.find(channel);
  if (iter == m_planes.end()) {
    return nullptr;
  }

  if (out_stride) {
    *out_stride = iter->second.stride;
  }

  return iter->second.mem;
}


const uint8_t* HeifPixelImage::get_plane(enum heif_channel channel, int* out_stride) const
{
  auto iter = m_planes.find(channel);
  if (iter == m_planes.end()) {
    return nullptr;
  }

  if (out_stride) {
    *out_stride = iter->second.stride;
  }

  return iter->second.mem;
}


void HeifPixelImage::copy_new_plane_from(const std::shared_ptr<const HeifPixelImage>& src_image,
                                         heif_channel src_channel,
                                         heif_channel dst_channel)
{
  int width = src_image->get_width(src_channel);
  int height = src_image->get_height(src_channel);

  assert(!has_channel(dst_channel));

  add_plane(dst_channel, width, height, src_image->get_bits_per_pixel(src_channel));

  uint8_t* dst;
  int dst_stride = 0;

  const uint8_t* src;
  int src_stride = 0;

  src = src_image->get_plane(src_channel, &src_stride);
  dst = get_plane(dst_channel, &dst_stride);

  int bpl = width * (src_image->get_storage_bits_per_pixel(src_channel) / 8);

  for (int y = 0; y < height; y++) {
    memcpy(dst + y * dst_stride, src + y * src_stride, bpl);
  }
}

void HeifPixelImage::fill_new_plane(heif_channel dst_channel, uint16_t value, int width, int height, int bpp)
{
  add_plane(dst_channel, width, height, bpp);

  int num_interleaved = num_interleaved_pixels_per_plane(m_chroma);

  if (bpp <= 8) {
    uint8_t* dst;
    int dst_stride = 0;
    dst = get_plane(dst_channel, &dst_stride);
    int width_bytes = width * num_interleaved;

    for (int y = 0; y < height; y++) {
      memset(dst + y * dst_stride, value, width_bytes);
    }
  }
  else {
    uint16_t* dst;
    int dst_stride = 0;
    dst = (uint16_t*) get_plane(dst_channel, &dst_stride);

    dst_stride /= 2;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width * num_interleaved; x++) {
        dst[y * dst_stride + x] = value;
      }
    }
  }
}


void HeifPixelImage::transfer_plane_from_image_as(const std::shared_ptr<HeifPixelImage>& source,
                                                  heif_channel src_channel,
                                                  heif_channel dst_channel)
{
  // TODO: check that dst_channel does not exist yet

  ImagePlane plane = source->m_planes[src_channel];
  source->m_planes.erase(src_channel);

  m_planes.insert(std::make_pair(dst_channel, plane));
}


bool is_chroma_with_alpha(heif_chroma chroma)
{
  switch (chroma) {
    case heif_chroma_undefined:
    case heif_chroma_monochrome:
    case heif_chroma_420:
    case heif_chroma_422:
    case heif_chroma_444:
    case heif_chroma_interleaved_RGB:
    case heif_chroma_interleaved_RRGGBB_BE:
    case heif_chroma_interleaved_RRGGBB_LE:
      return false;

    case heif_chroma_interleaved_RGBA:
    case heif_chroma_interleaved_RRGGBBAA_BE:
    case heif_chroma_interleaved_RRGGBBAA_LE:
      return true;
  }

  assert(false);
  return false;
}


Error HeifPixelImage::rotate_ccw(int angle_degrees,
                                 std::shared_ptr<HeifPixelImage>& out_img)
{
  // --- create output image (or simply reuse existing image)

  if (angle_degrees == 0) {
    out_img = shared_from_this();
    return Error::Ok;
  }

  int out_width = m_width;
  int out_height = m_height;

  if (angle_degrees == 90 || angle_degrees == 270) {
    std::swap(out_width, out_height);
  }

  out_img = std::make_shared<HeifPixelImage>();
  out_img->create(out_width, out_height, m_colorspace, m_chroma);


  // --- rotate all channels

  for (const auto& plane_pair : m_planes) {
    heif_channel channel = plane_pair.first;
    const ImagePlane& plane = plane_pair.second;

    /*
    if (plane.bit_depth != 8) {
      return Error(heif_error_Unsupported_feature,
                   heif_suberror_Unspecified,
                   "Can currently only rotate images with 8 bits per pixel");
    }
    */

    int out_plane_width = plane.m_width;
    int out_plane_height = plane.m_height;

    if (angle_degrees == 90 || angle_degrees == 270) {
      std::swap(out_plane_width, out_plane_height);
    }

    out_img->add_plane(channel, out_plane_width, out_plane_height, plane.m_bit_depth);


    int w = plane.m_width;
    int h = plane.m_height;

    int in_stride = plane.stride;
    const uint8_t* in_data = plane.mem;

    int out_stride = 0;
    uint8_t* out_data = out_img->get_plane(channel, &out_stride);

    if (plane.m_bit_depth == 8) {

      int convert_result = -1;

      if((this->m_colorspace == heif_colorspace_RGB) && (this->m_chroma == heif_chroma_interleaved_RGBA))
      {
#if HAVE_YUV
        libyuv::RotationModeEnum mode ;
        switch(angle_degrees) 
        {
          case 0  : mode = libyuv::kRotate0  ; break;
          case 90 : mode = libyuv::kRotate270; break;
          case 180: mode = libyuv::kRotate180; break;
          case 270: mode = libyuv::kRotate90 ; break;
        }
        convert_result = libyuv::ARGBRotate(in_data, in_stride, out_data, out_stride, w, h, mode);
#endif
      }

      if(convert_result < 0 ) 
      {
        if (angle_degrees == 270) {
          for (long long x = 0; x < h; x++)
            for (long long y = 0; y < w; y++) {
              if(m_chroma >= heif_chroma_interleaved_RGB ) {
                int channel_num = (m_chroma == heif_chroma_interleaved_RGB) ? 3 : ((m_chroma == heif_chroma_interleaved_RGBA) ? 4 : 1) ;
                for(int channel = 0; channel < channel_num; channel++)
                  out_data[y * out_stride + x*channel_num + channel] = in_data[(h - 1 - x) * in_stride + y*channel_num + channel];
              } 
              else {
                out_data[y * out_stride + x] = in_data[(h - 1 - x) * in_stride + y];
              }
            }
        }
        else if (angle_degrees == 180) {
          for (long long y = 0; y < h; y++)
            for (long long x = 0; x < w; x++) {
              if(m_chroma >= heif_chroma_interleaved_RGB ) {
                int channel_num = (m_chroma == heif_chroma_interleaved_RGB) ? 3 : ((m_chroma == heif_chroma_interleaved_RGBA) ? 4 : 1) ;
                for(int channel = 0; channel < channel_num; channel++)
                  out_data[y * out_stride + x*channel_num + channel] = in_data[(h - 1 - y) * in_stride + (w - 1 - x)*channel_num + channel];
              } 
              else {
                out_data[y * out_stride + x] = in_data[(h - 1 - y) * in_stride + (w - 1 - x)];
              }
            }
        }
        else if (angle_degrees == 90) {
          for (long x = 0; x < h; x++)
            for (long y = 0; y < w; y++) {
              if(m_chroma >= heif_chroma_interleaved_RGB ) {
                int channel_num = (m_chroma == heif_chroma_interleaved_RGB) ? 3 : ((m_chroma == heif_chroma_interleaved_RGBA) ? 4 : 1) ;
                for(int channel = 0; channel < channel_num; channel++)
                  out_data[y * out_stride + x*channel_num + channel] = in_data[x * in_stride + (w - 1 - y)*channel_num + channel];
              } 
              else {
                out_data[y * out_stride + x] = in_data[x * in_stride + (w - 1 - y)];
              }
            }
          }
      }
    }
    else { // 16 bit (TODO: unchecked code)
      if (angle_degrees == 270) {
        for (long long x = 0; x < h; x++)
          for (long long y = 0; y < w; y++) {
            if(m_chroma >= heif_chroma_interleaved_RRGGBB_BE ) {
              int channel_num = 1;
              if((m_chroma == heif_chroma_interleaved_RRGGBB_BE) || (m_chroma == heif_chroma_interleaved_RRGGBB_LE)) { 
                channel_num = 3;
              } 
              else if((m_chroma == heif_chroma_interleaved_RRGGBBAA_BE) || (m_chroma == heif_chroma_interleaved_RRGGBBAA_LE)) {
                channel_num = 4;
              }
              else {
                channel_num = 1;
              }
              for(int channel = 0; channel < channel_num; channel++) {
                out_data[y * out_stride + 2 * x * channel_num + 2 * channel]     = in_data[(h - 1 - x) * in_stride + 2 * y * channel_num + 2 * channel];
                out_data[y * out_stride + 2 * x * channel_num + 2 * channel + 1] = in_data[(h - 1 - x) * in_stride + 2 * y * channel_num + 2 * channel + 1];
              }
            } 
            else {
              out_data[y * out_stride + 2 * x]     = in_data[(h - 1 - x) * in_stride + 2 * y];
              out_data[y * out_stride + 2 * x + 1] = in_data[(h - 1 - x) * in_stride + 2 * y + 1];
            }
          }
      }
      else if (angle_degrees == 180) {
        for (long long y = 0; y < h; y++)
          for (long long x = 0; x < w; x++) {
            if(m_chroma >= heif_chroma_interleaved_RRGGBB_BE ) {
              int channel_num = 1;
              if((m_chroma == heif_chroma_interleaved_RRGGBB_BE) || (m_chroma == heif_chroma_interleaved_RRGGBB_LE)) { 
                channel_num = 3;
              } 
              else if((m_chroma == heif_chroma_interleaved_RRGGBBAA_BE) || (m_chroma == heif_chroma_interleaved_RRGGBBAA_LE)) {
                channel_num = 4;
              }
              else {
                channel_num = 1;
              }
              for(int channel = 0; channel < channel_num; channel++) {
                out_data[y * out_stride + 2 * x * channel_num + 2 * channel]     = in_data[(h - 1 - y) * in_stride + 2 * (w - 1 - x) * channel_num + 2 * channel];
                out_data[y * out_stride + 2 * x * channel_num + 2 * channel + 1] = in_data[(h - 1 - y) * in_stride + 2 * (w - 1 - x) * channel_num + 2 * channel + 1];
              }
            } 
            else {
              out_data[y * out_stride + 2 * x]     = in_data[(h - 1 - y) * in_stride + 2 * (w - 1 - x)];
              out_data[y * out_stride + 2 * x + 1] = in_data[(h - 1 - y) * in_stride + 2 * (w - 1 - x) + 1];
            }
          }
      }
      else if (angle_degrees == 90) {
        for (long long x = 0; x < h; x++)
          for (long long y = 0; y < w; y++) {
            if(m_chroma >= heif_chroma_interleaved_RRGGBB_BE ) {
              int channel_num = 1;
              if((m_chroma == heif_chroma_interleaved_RRGGBB_BE) || (m_chroma == heif_chroma_interleaved_RRGGBB_LE)) { 
                channel_num = 3;
              } 
              else if((m_chroma == heif_chroma_interleaved_RRGGBBAA_BE) || (m_chroma == heif_chroma_interleaved_RRGGBBAA_LE)) {
                channel_num = 4;
              }
              else {
                channel_num = 1;
              }
              for(int channel = 0; channel < channel_num; channel++) {
                out_data[y * out_stride + 2 * x * channel_num + 2 * channel]     = in_data[x * in_stride + 2 * (w - 1 - y) * channel_num + 2 * channel];
                out_data[y * out_stride + 2 * x * channel_num + 2 * channel + 1] = in_data[x * in_stride + 2 * (w - 1 - y) * channel_num + 2 * channel + 1];
              }
            } 
            else {
              out_data[y * out_stride + 2 * x]     = in_data[x * in_stride + 2 * (w - 1 - y)];
              out_data[y * out_stride + 2 * x + 1] = in_data[x * in_stride + 2 * (w - 1 - y) + 1];
            }
          }
      }
    }
  }

  // --- pass the color profiles to the new image

  out_img->set_color_profile_nclx(get_color_profile_nclx());
  out_img->set_color_profile_icc(get_color_profile_icc());

  return Error::Ok;
}


Error HeifPixelImage::mirror_inplace(heif_transform_mirror_direction direction)
{
  for (auto& plane_pair : m_planes) {
    ImagePlane& plane = plane_pair.second;

    if (plane.m_bit_depth != 8) {
      return Error(heif_error_Unsupported_feature,
                   heif_suberror_Unspecified,
                   "Can currently only mirror images with 8 bits per pixel");
    }


    int w = plane.m_width;
    int h = plane.m_height;

    int stride = plane.stride;
    uint8_t* data = plane.mem;

    if((this->m_colorspace == heif_colorspace_RGB) && (this->m_chroma == heif_chroma_interleaved_RGBA))
    {
      if (direction == heif_transform_mirror_direction_horizontal) {
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < 4 * w / 2; x++)
            std::swap(data[y * stride + x], data[y * stride + 4 * w - 1 - x]);
        }
      }
      else {
        for (int y = 0; y < h / 2; y++) {
          for (int x = 0; x < 4 * w; x++)
            std::swap(data[y * stride + x], data[(h - 1 - y) * stride + x]);
        }
      }
    }
    else 
    {
      if (direction == heif_transform_mirror_direction_horizontal) {
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w / 2; x++)
            std::swap(data[y * stride + x], data[y * stride + w - 1 - x]);
        }
      }
      else {
        for (int y = 0; y < h / 2; y++) {
          for (int x = 0; x < w; x++)
            std::swap(data[y * stride + x], data[(h - 1 - y) * stride + x]);
        }
      }
    }
  }

  return Error::Ok;
}


Error HeifPixelImage::crop(int left, int right, int top, int bottom,
                           std::shared_ptr<HeifPixelImage>& out_img) const
{
  out_img = std::make_shared<HeifPixelImage>();
  out_img->create(right - left + 1, bottom - top + 1, m_colorspace, m_chroma);


  // --- crop all channels

  for (const auto& plane_pair : m_planes) {
    heif_channel channel = plane_pair.first;
    const ImagePlane& plane = plane_pair.second;

    if (false && plane.m_bit_depth != 8) {
      return Error(heif_error_Unsupported_feature,
                   heif_suberror_Unspecified,
                   "Can currently only crop images with 8 bits per pixel");
    }


    int w = plane.m_width;
    int h = plane.m_height;

    int plane_left = left * w / m_width;
    int plane_right = right * w / m_width;
    int plane_top = top * h / m_height;
    int plane_bottom = bottom * h / m_height;

    if((this->m_colorspace == heif_colorspace_RGB) && (this->m_chroma == heif_chroma_interleaved_RGBA))
    {
      out_img->add_plane(channel,
                         (plane_right - plane_left + 1)*4,
                         plane_bottom - plane_top + 1,
                         plane.m_bit_depth);
    }
    else 
    {
      out_img->add_plane(channel,
                         plane_right - plane_left + 1,
                         plane_bottom - plane_top + 1,
                         plane.m_bit_depth);
    }


    int in_stride = plane.stride;
    const uint8_t* in_data = plane.mem;

    int out_stride = 0;
    uint8_t* out_data = out_img->get_plane(channel, &out_stride);

    if((this->m_colorspace == heif_colorspace_RGB) && (this->m_chroma == heif_chroma_interleaved_RGBA))
    {
      if (plane.m_bit_depth == 8) {
        for (int y = plane_top; y <= plane_bottom; y++) {
          memcpy(&out_data[(y - plane_top) * out_stride],
                 &in_data[y * in_stride + plane_left * 4],
                 (plane_right - plane_left + 1) * 4 );
        }
      }
      else {
        for (int y = plane_top; y <= plane_bottom; y++) {
          memcpy(&out_data[(y - plane_top) * out_stride],
                 &in_data[y * in_stride + plane_left * 8],
                 (plane_right - plane_left + 1) * 2 * 4);
        }
      }
    }
    else 
    {
      if (plane.m_bit_depth == 8) {
        for (int y = plane_top; y <= plane_bottom; y++) {
          memcpy(&out_data[(y - plane_top) * out_stride],
                 &in_data[y * in_stride + plane_left],
                 plane_right - plane_left + 1);
        }
      }
      else {
        for (int y = plane_top; y <= plane_bottom; y++) {
          memcpy(&out_data[(y - plane_top) * out_stride],
                 &in_data[y * in_stride + plane_left * 2],
                 (plane_right - plane_left + 1) * 2);
        }
      }
    }
  }

  // --- pass the color profiles to the new image

  out_img->set_color_profile_nclx(get_color_profile_nclx());
  out_img->set_color_profile_icc(get_color_profile_icc());

  return Error::Ok;
}


#define PREMULTI_PIXEL(dst, src, alpha) \
{ \
  uint32_t src_in = src;  \
  uint32_t alpha_in = alpha; \
  uint32_t tmp = (src_in * alpha_in + 128) >> 8; \
  dst = uint8_t(tmp); \
}

Error HeifPixelImage::rgba_premultiply_alpha(void )
{

  int width = this->m_width ;
  int height= this->m_height;
  int bpp   = this->get_bits_per_pixel(heif_channel_interleaved);

  if(!this->has_channel(heif_channel_interleaved) ) {
    return Error(heif_error_Usage_error,
                heif_suberror_Unsupported_image_type);
  }
  if(this->get_colorspace() == heif_colorspace_undefined ||
     this->get_colorspace() == heif_colorspace_YCbCr ) {
    return Error(heif_error_Usage_error,
                heif_suberror_Unsupported_image_type);
  }

  uint8_t* rgba_p;
  int rgba_p_stride = 0;

  rgba_p = (uint8_t*) this->get_plane(heif_channel_interleaved, &rgba_p_stride);

#if HAVE_YUV
  if(libyuv::ARGBAttenuate(rgba_p, rgba_p_stride, rgba_p, rgba_p_stride, width, height) != 0) {
    return Error(heif_error_Usage_error,
                heif_suberror_Unsupported_image_type);
  }
#else
  uint8_t * rgba_row = rgba_p;
  for(int h = 0; h < height; h++)
  {
    for(int w = 0 ; w < width; w++) 
    {
      PREMULTI_PIXEL(*(rgba_row + 4 * w + 0), *(rgba_row + 4 * w + 0), *(rgba_row + 4 * w + 3));
      PREMULTI_PIXEL(*(rgba_row + 4 * w + 1), *(rgba_row + 4 * w + 1), *(rgba_row + 4 * w + 3));
      PREMULTI_PIXEL(*(rgba_row + 4 * w + 2), *(rgba_row + 4 * w + 2), *(rgba_row + 4 * w + 3));
    }
    rgba_row += rgba_p_stride  ;
  }
#endif

  this->set_premultiplied_alpha(true);

  return Error::Ok;
}


Error HeifPixelImage::fill_RGB_16bit(uint16_t r, uint16_t g, uint16_t b, uint16_t a)
{
  for (const auto& channel : {heif_channel_R, heif_channel_G, heif_channel_B, heif_channel_Alpha}) {

    const auto plane_iter = m_planes.find(channel);
    if (plane_iter == m_planes.end()) {

      // alpha channel is optional, R,G,B is required
      if (channel == heif_channel_Alpha) {
        continue;
      }

      return Error(heif_error_Usage_error,
                   heif_suberror_Nonexisting_image_channel_referenced);

    }

    ImagePlane& plane = plane_iter->second;

    if (plane.m_bit_depth != 8) {
      return Error(heif_error_Unsupported_feature,
                   heif_suberror_Unspecified,
                   "Can currently only fill images with 8 bits per pixel");
    }

    size_t h = plane.m_height;

    size_t stride = plane.stride;
    uint8_t* data = plane.mem;

    uint16_t val16;
    switch (channel) {
      case heif_channel_R:
        val16 = r;
        break;
      case heif_channel_G:
        val16 = g;
        break;
      case heif_channel_B:
        val16 = b;
        break;
      case heif_channel_Alpha:
        val16 = a;
        break;
      default:
        // initialization only to avoid warning of uninitialized variable.
        val16 = 0;
        // Should already be detected by the check above ("m_planes.find").
        assert(false);
    }

    auto val8 = static_cast<uint8_t>(val16 >> 8U);


    // memset() even when h * stride > sizeof(size_t)

    if (std::numeric_limits<size_t>::max() / stride > h) {
      // can fill in one step
      memset(data, val8, stride * h);
    }
    else {
      // fill line by line
      auto* p = data;

      for (size_t y=0;y<h;y++) {
        memset(p, val8, stride);
        p += stride;
      }
    }
  }

  return Error::Ok;
}


uint32_t negate_negative_int32(int32_t x)
{
  assert(x <= 0);

  if (x == INT32_MIN) {
    return static_cast<uint32_t>(INT32_MAX) + 1;
  }
  else {
    return static_cast<uint32_t>(-x);
  }
}


Error HeifPixelImage::overlay(std::shared_ptr<HeifPixelImage>& overlay, int32_t dx, int32_t dy)
{
  std::set<enum heif_channel> channels = overlay->get_channel_set();

  bool has_alpha = overlay->has_channel(heif_channel_Alpha);
  //bool has_alpha_me = has_channel(heif_channel_Alpha);

  int alpha_stride = 0;
  uint8_t* alpha_p;
  alpha_p = overlay->get_plane(heif_channel_Alpha, &alpha_stride);

  for (heif_channel channel : channels) {
    if (!has_channel(channel)) {
      continue;
    }

    int in_stride = 0;
    const uint8_t* in_p;

    int out_stride = 0;
    uint8_t* out_p;

    in_p = overlay->get_plane(channel, &in_stride);
    out_p = get_plane(channel, &out_stride);

    uint32_t in_w = overlay->get_width(channel);
    uint32_t in_h = overlay->get_height(channel);

    uint32_t out_w = get_width(channel);
    uint32_t out_h = get_height(channel);

    // top-left points where to start copying in source and destination
    uint32_t in_x0;
    uint32_t in_y0;
    uint32_t out_x0;
    uint32_t out_y0;

    if (dx > 0 && static_cast<uint32_t>(dx) >= out_w) {
      // the overlay image is completely outside the right border -> skip overlaying
      return Error::Ok;
    }
    else if (dx < 0 && in_w <= negate_negative_int32(dx)) {
      // the overlay image is completely outside the left border -> skip overlaying
      return Error::Ok;
    }

    if (dx < 0) {
      // overlay image started partially outside of left border

      in_x0 = negate_negative_int32(dx);
      out_x0 = 0;
      in_w = in_w - in_x0; // in_x0 < in_w because in_w > -dx = in_x0
    }
    else {
      in_x0 = 0;
      out_x0 = static_cast<uint32_t>(dx);
    }

    // we know that dx >= 0 && dx < out_w

    if (static_cast<uint32_t>(dx) > UINT32_MAX - in_w ||
        dx + in_w > out_w) {
      // overlay image extends partially outside of right border

      in_w = out_w - static_cast<uint32_t>(dx); // we know that dx < out_w from first condition
    }


    if (dy > 0 && static_cast<uint32_t>(dy) >= out_h) {
      // the overlay image is completely outside the bottom border -> skip overlaying
      return Error::Ok;
    }
    else if (dy < 0 && in_h <= negate_negative_int32(dy)) {
      // the overlay image is completely outside the top border -> skip overlaying
      return Error::Ok;
    }

    if (dy < 0) {
      // overlay image started partially outside of top border

      in_y0 = negate_negative_int32(dy);
      out_y0 = 0;
      in_h = in_h - in_y0; // in_y0 < in_h because in_h > -dy = in_y0
    }
    else {
      in_y0 = 0;
      out_y0 = static_cast<uint32_t>(dy);
    }

    // we know that dy >= 0 && dy < out_h

    if (static_cast<uint32_t>(dy) > UINT32_MAX - in_h ||
        dy + in_h > out_h) {
      // overlay image extends partially outside of bottom border

      in_h = out_h - static_cast<uint32_t>(dy); // we know that dy < out_h from first condition
    }


    for (uint32_t y = in_y0; y < in_h; y++) {
      if (!has_alpha) {
        memcpy(out_p + out_x0 + (out_y0 + y - in_y0) * out_stride,
               in_p + in_x0 + y * in_stride,
               in_w - in_x0);
      }
      else {
        for (uint32_t x = in_x0; x < in_w; x++) {
          uint8_t* outptr = &out_p[out_x0 + (out_y0 + y - in_y0) * out_stride + x];
          uint8_t in_val = in_p[in_x0 + y * in_stride + x];
          uint8_t alpha_val = alpha_p[in_x0 + y * in_stride + x];

          *outptr = (uint8_t) ((in_val * alpha_val + *outptr * (255 - alpha_val)) / 255);
        }
      }
    }
  }

  return Error::Ok;
}


Error HeifPixelImage::scale_nearest_neighbor(std::shared_ptr<HeifPixelImage>& out_img,
                                             int width, int height) const
{
  out_img = std::make_shared<HeifPixelImage>();
  out_img->create(width, height, m_colorspace, m_chroma);


  // --- create output image with scaled planes

  if (has_channel(heif_channel_interleaved)) {
    out_img->add_plane(heif_channel_interleaved, width, height, get_bits_per_pixel(heif_channel_interleaved));
  }
  else {
    if (get_colorspace() == heif_colorspace_RGB) {
      if (!has_channel(heif_channel_R) ||
          !has_channel(heif_channel_G) ||
          !has_channel(heif_channel_B)) {
        return Error(heif_error_Invalid_input, heif_suberror_Unspecified, "RGB input without R,G,B, planes");
      }

      out_img->add_plane(heif_channel_R, width, height, get_bits_per_pixel(heif_channel_R));
      out_img->add_plane(heif_channel_G, width, height, get_bits_per_pixel(heif_channel_G));
      out_img->add_plane(heif_channel_B, width, height, get_bits_per_pixel(heif_channel_B));
    }
    else if (get_colorspace() == heif_colorspace_monochrome) {
      if (!has_channel(heif_channel_Y)) {
        return Error(heif_error_Invalid_input, heif_suberror_Unspecified, "monochrome input with no Y plane");
      }

      out_img->add_plane(heif_channel_Y, width, height, get_bits_per_pixel(heif_channel_Y));
    }
    else if (get_colorspace() == heif_colorspace_YCbCr) {
      if (!has_channel(heif_channel_Y) ||
          !has_channel(heif_channel_Cb) ||
          !has_channel(heif_channel_Cr)) {
        return Error(heif_error_Invalid_input, heif_suberror_Unspecified, "YCbCr image without Y,Cb,Cr planes");
      }

      int cw, ch;
      get_subsampled_size(width, height, heif_channel_Cb, get_chroma_format(), &cw, &ch);
      out_img->add_plane(heif_channel_Y, width, height, get_bits_per_pixel(heif_channel_Y));
      out_img->add_plane(heif_channel_Cb, cw, ch, get_bits_per_pixel(heif_channel_Cb));
      out_img->add_plane(heif_channel_Cr, cw, ch, get_bits_per_pixel(heif_channel_Cr));
    }
    else {
      return Error(heif_error_Invalid_input, heif_suberror_Unspecified, "unknown color configuration");
    }

    if (has_channel(heif_channel_Alpha)) {
      out_img->add_plane(heif_channel_Alpha, width, height, get_bits_per_pixel(heif_channel_Alpha));
    }
  }


  // --- scale all channels

  for (const auto& plane_pair : m_planes) {
    heif_channel channel = plane_pair.first;
    const ImagePlane& plane = plane_pair.second;

    const int bpp = get_storage_bits_per_pixel(channel) / 8;

    if (!out_img->has_channel(channel)) {
      return Error(heif_error_Invalid_input, heif_suberror_Unspecified, "scaling input has extra color plane");
    }

    int out_w = out_img->get_width(channel);
    int out_h = out_img->get_height(channel);

    int in_stride = plane.stride;
    const uint8_t* in_data = plane.mem;

    int out_stride = 0;
    uint8_t* out_data = out_img->get_plane(channel, &out_stride);


    for (int y = 0; y < out_h; y++) {
      int iy = y * m_height / height;

      if (bpp == 1) {
        for (int x = 0; x < out_w; x++) {
          int ix = x * m_width / width;

          out_data[y * out_stride + x] = in_data[iy * in_stride + ix];
        }
      }
      else {
        for (int x = 0; x < out_w; x++) {
          int ix = x * m_width / width;

          for (int b = 0; b < bpp; b++) {
            out_data[y * out_stride + bpp * x + b] = in_data[iy * in_stride + bpp * ix + b];
          }
        }
      }
    }
  }

  return Error::Ok;
}


void HeifPixelImage::debug_dump() const
{
  auto channels = get_channel_set();
  for (auto c : channels) {
    int stride = 0;
    const uint8_t* p = get_plane(c, &stride);

    for (int y = 0; y < 8; y++) {
      for (int x = 0; x < 8; x++) {
        printf("%02x ", p[y * stride + x]);
      }
      printf("\n");
    }
  }
}
