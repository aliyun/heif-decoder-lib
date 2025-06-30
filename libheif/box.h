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

#ifndef LIBHEIF_BOX_H
#define LIBHEIF_BOX_H

#include <cstdint>
#include "common_utils.h"
#include "libheif/heif.h"
#include "libheif/heif_properties.h"
#include <cinttypes>
#include <cstddef>

#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <istream>
#include <bitset>
#include <utility>

#include "error.h"
#include "logging.h"
#include "bitstream.h"

#if !defined(__EMSCRIPTEN__) && !defined(_MSC_VER)
// std::array<bool> is not supported on some older compilers.
#define HAS_BOOL_ARRAY 1
#endif

// abbreviation
constexpr inline uint32_t fourcc(const char* id) { return fourcc_to_uint32(id); }

std::string to_fourcc(uint32_t code);

/*
  constexpr uint32_t fourcc(const char* string)
  {
  return ((string[0]<<24) |
  (string[1]<<16) |
  (string[2]<< 8) |
  (string[3]));
  }
*/

struct sampleofchunk {
  uint32_t m_first_chunk;
  uint32_t m_samples_per_chunk;
  uint32_t m_sample_description_index;
};

class Fraction
{
public:
  Fraction() = default;

  Fraction(int32_t num, int32_t den);

  // may only use values up to int32_t maximum
  Fraction(uint32_t num, uint32_t den);

  // Values will be reduced until they fit into int32_t.
  Fraction(int64_t num, int64_t den);

  Fraction operator+(const Fraction&) const;

  Fraction operator-(const Fraction&) const;

  Fraction operator+(int) const;

  Fraction operator-(int) const;

  Fraction operator/(int) const;

  int32_t round_down() const;

  int32_t round_up() const;

  int32_t round() const;

  bool is_valid() const;

  double to_double() const {
    return numerator / (double)denominator;
  }

  int32_t numerator = 0;
  int32_t denominator = 1;
};


inline std::ostream& operator<<(std::ostream& str, const Fraction& f)
{
  str << f.numerator << "/" << f.denominator;
  return str;
}


class BoxHeader
{
public:
  BoxHeader();

  virtual ~BoxHeader() = default;

  constexpr static uint64_t size_until_end_of_file = 0;

  uint64_t get_box_size() const { return m_size; }

  bool has_fixed_box_size() const { return m_size != 0; }

  uint32_t get_header_size() const { return m_header_size; }

  uint32_t get_short_type() const { return m_type; }

  std::vector<uint8_t> get_type() const;

  std::string get_type_string() const;

  void set_short_type(uint32_t type) { m_type = type; }


  // should only be called if get_short_type == fourcc("uuid")
  std::vector<uint8_t> get_uuid_type() const;

  void set_uuid_type(const std::vector<uint8_t>&);


  Error parse_header(BitstreamRange& range);

  virtual std::string dump(Indent&) const;


  virtual bool is_full_box_header() const { return false; }


private:
  uint64_t m_size = 0;

  uint32_t m_type = 0;
  std::vector<uint8_t> m_uuid_type;

protected:
  uint32_t m_header_size = 0;
};


class Box : public BoxHeader
{
public:
  Box() = default;

  void set_short_header(const BoxHeader& hdr)
  {
    *(BoxHeader*) this = hdr;
  }

  // header size without the FullBox fields (if applicable)
  int calculate_header_size(bool data64bit) const;

  static Error read(BitstreamRange& range, std::shared_ptr<Box>* box);

  virtual Error write(StreamWriter& writer) const;

  // check, which box version is required and set this in the (full) box header
  virtual void derive_box_version() {}

  void derive_box_version_recursive();

  std::string dump(Indent&) const override;

  std::shared_ptr<Box> get_child_box(uint32_t short_type) const;

  std::vector<std::shared_ptr<Box>> get_child_boxes(uint32_t short_type) const;

  template<typename T> [[nodiscard]] std::shared_ptr<T> get_child_box() const
  {
    // TODO: we could remove the dynamic_cast<> by adding the fourcc type of each Box
    //       as a "constexpr uint32_t Box::short_type", compare to that and use static_cast<>
    for (auto& box : m_children) {
      if (auto typed_box = std::dynamic_pointer_cast<T>(box)) {
        return typed_box;
      }
    }

    return nullptr;
  }

  template<typename T>
  std::vector<std::shared_ptr<T>> get_child_boxes() const
  {
    std::vector<std::shared_ptr<T>> result;
    for (auto& box : m_children) {
      if (auto typed_box = std::dynamic_pointer_cast<T>(box)) {
        result.push_back(typed_box);
      }
    }

    return result;
  }


  template<typename T>
  std::vector<std::shared_ptr<T>> get_typed_child_boxes(uint32_t short_type) const
  {
    auto boxes = get_child_boxes(short_type);
    std::vector<std::shared_ptr<T>> typedBoxes;
    for (const auto& box : boxes) {
      typedBoxes.push_back(std::dynamic_pointer_cast<T>(box));
    }
    return typedBoxes;
  }

  const std::vector<std::shared_ptr<Box>>& get_all_child_boxes() const { return m_children; }

  int append_child_box(const std::shared_ptr<Box>& box)
  {
    m_children.push_back(box);
    return (int) m_children.size() - 1;
  }

  virtual bool operator==(const Box& other) const;

  static bool equal(const std::shared_ptr<Box>& box1, const std::shared_ptr<Box>& box2);

  void set_output_position(uint64_t pos) { m_output_position = pos; }

protected:
  virtual Error parse(BitstreamRange& range);

  std::vector<std::shared_ptr<Box>> m_children;

  const static int READ_CHILDREN_ALL = -1;

  const static uint64_t INVALID_POSITION = 0xFFFFFFFFFFFFFFFF;

  uint64_t m_input_position = INVALID_POSITION;
  uint64_t m_output_position = INVALID_POSITION;

  Error read_children(BitstreamRange& range, int number = READ_CHILDREN_ALL);

  Error write_children(StreamWriter& writer) const;

  std::string dump_children(Indent&) const;


  // --- writing

  virtual size_t reserve_box_header_space(StreamWriter& writer, bool data64bit = false) const;

  Error prepend_header(StreamWriter&, size_t box_start, bool data64bit = false) const;

  virtual Error write_header(StreamWriter&, size_t total_box_size, bool data64bit = false) const;
};


class FullBox : public Box
{
public:
  bool is_full_box_header() const override { return true; }

  std::string dump(Indent& indent) const override;

  void derive_box_version() override { set_version(0); }


  Error parse_full_box_header(BitstreamRange& range);

  uint8_t get_version() const { return m_version; }

  void set_version(uint8_t version) { m_version = version; }

  uint32_t get_flags() const { return m_flags; }

  void set_flags(uint32_t flags) { m_flags = flags; }

protected:

  // --- writing

  size_t reserve_box_header_space(StreamWriter& writer, bool data64bit = false) const override;

  Error write_header(StreamWriter&, size_t total_size, bool data64bit = false) const override;

  Error unsupported_version_error(const char* box) const;

private:
  uint8_t m_version = 0;
  uint32_t m_flags = 0;
};


class Box_other : public Box
{
public:
  Box_other(uint32_t short_type)
  {
    set_short_type(short_type);
  }

  const std::vector<uint8_t>& get_raw_data() const { return m_data; }

  void set_raw_data(const std::vector<uint8_t>& data) { m_data = data; }

  Error write(StreamWriter& writer) const override;

  std::string dump(Indent&) const override;

protected:
  Error parse(BitstreamRange& range) override;

  std::vector<uint8_t> m_data;
};




class Box_ftyp : public Box
{
public:
  Box_ftyp()
  {
    set_short_type(fourcc("ftyp"));
  }

  std::string dump(Indent&) const override;

  bool has_compatible_brand(uint32_t brand) const;

  std::vector<uint32_t> list_brands() const { return m_compatible_brands; }

  void set_major_brand(heif_brand2 major_brand) { m_major_brand = major_brand; }

  void set_minor_version(uint32_t minor_version) { m_minor_version = minor_version; }

  void clear_compatible_brands() { m_compatible_brands.clear(); }

  void add_compatible_brand(heif_brand2 brand);

  Error write(StreamWriter& writer) const override;
  
  bool get_compatiable_bands_avaliable() {return compatiable_bands_avaliable;}
    
  void set_compatiable_bands_avaliable(bool avaliable) {compatiable_bands_avaliable = avaliable;}

protected:
  Error parse(BitstreamRange& range) override;

private:
  bool compatiable_bands_avaliable = false;
  uint32_t m_major_brand = 0;
  uint32_t m_minor_version = 0;
  std::vector<heif_brand2> m_compatible_brands;
};


class Box_meta : public FullBox
{
public:
  Box_meta()
  {
    set_short_type(fourcc("meta"));
  }

  std::string dump(Indent&) const override;

protected:
  Error parse(BitstreamRange& range) override;
};

  class Box_moov : public Box
  {
  public:
    Box_moov()
    {
      set_short_type(fourcc("moov"));
    }

    std::string dump(Indent&) const override;
      
  protected:
    Error parse(BitstreamRange& range) override;
  };
  
  class Box_mvhd : public FullBox
  {
  public:
    Box_mvhd()
    {
      set_short_type(fourcc("mvhd"));
    }

    void derive_box_version() override;

    void set_mvhd_data(uint64_t creation_time, uint64_t modification_time, uint32_t timescale, uint64_t duration,
                       uint32_t next_track_ID)
    {
      m_creation_time = creation_time;
      m_modification_time = modification_time;
      m_timescale = timescale;
      m_duration = duration;
      m_next_track_ID = next_track_ID;
    }
    
    uint32_t get_timescale()
    {
      return m_timescale;
    };
    
    uint64_t get_duration()
    {
      return m_duration;
    };

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    uint64_t m_creation_time;
    uint64_t m_modification_time;
    uint32_t m_timescale;
    uint64_t m_duration;
    uint32_t m_matrix[9] = {0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000};
    uint32_t m_next_track_ID;
    uint32_t m_rate;
    uint16_t m_volume;
    uint16_t m_reserved0 = 0;
    uint32_t m_reserved1[2] = {0,};
    uint32_t pre_defined[6] = {0,};
  };

  class Box_trak : public Box
  {
  public:
    Box_trak()
    {
      set_short_type(fourcc("trak"));
    }

    std::string dump(Indent&) const override;
      
  protected:
    Error parse(BitstreamRange& range) override;    
  };

  class Box_tkhd : public FullBox
  {
  public:
    Box_tkhd()
    {
      set_short_type(fourcc("tkhd"));
    }

    void derive_box_version() override;

    void set_tkhd_data(uint64_t creation_time, uint64_t modification_time, uint32_t track_ID, uint64_t duration,
                       uint32_t width, uint32_t height)
    {
      m_creation_time = creation_time;
      m_modification_time = modification_time;
      m_track_ID = track_ID;
      m_duration = duration;
      m_width = width;
      m_height = height;
    }

    std::string dump(Indent&) const override;

    uint32_t get_width() {return m_width;}

    uint32_t get_height() {return m_height;}

    Error write(StreamWriter& writer) const override;

  protected:
    Error parse(BitstreamRange& range) override;
    
  private:
    uint64_t m_creation_time;
    uint64_t m_modification_time;
    uint32_t m_track_ID;
    uint32_t m_reserved0 = 0;
    uint64_t m_duration;
    uint32_t m_reserved1[2] = {0,};
    uint16_t m_layer = 0;
    uint16_t m_alternate_group = 0;
    uint16_t m_volume = 0;
    uint16_t m_reserved2 = 0;
    uint32_t m_matrix[9] = {0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000};
    uint32_t m_width;
    uint32_t m_height;
  };

  class Box_mdia : public Box
  {
  public:
    Box_mdia()
    {
      set_short_type(fourcc("mdia"));
    }

    std::string dump(Indent&) const override;
      
  protected:
    Error parse(BitstreamRange& range) override;    
  };

  class Box_mdhd : public FullBox
  {
  public:
    Box_mdhd()
    {
      set_short_type(fourcc("mdhd"));
    }

    void derive_box_version() override;

    void set_mdhd_data(uint64_t creation_time, uint64_t modification_time, uint32_t timescale, uint64_t duration, uint16_t language)
    {
      m_creation_time = creation_time;
      m_modification_time = modification_time;
      m_timescale = timescale;
      m_duration = duration;
      m_language = language;
    }

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    uint64_t m_creation_time;
    uint64_t m_modification_time;
    uint32_t m_timescale;
    uint64_t m_duration;
    uint16_t m_language;
    uint16_t m_pre_defined = 0;
  };

  class Box_minf : public Box
  {
  public:
    Box_minf()
    {
      set_short_type(fourcc("minf"));
    }

    std::string dump(Indent&) const override;
      
  protected:
    Error parse(BitstreamRange& range) override;    
  };

  class Box_vmhd : public FullBox
  {
  public:
    Box_vmhd()
    {
      set_short_type(fourcc("vmhd"));
    }

    void derive_box_version() override;

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    uint16_t m_graphicsmode;
    uint16_t m_opcolor[3];
  };

  class Box_stbl : public Box
  {
  public:
    Box_stbl()
    {
      set_short_type(fourcc("stbl"));
    }

    std::string dump(Indent&) const override;
      
  protected:
    Error parse(BitstreamRange& range) override;    
  };

  class Box_stsd : public FullBox
  {
  public:
    Box_stsd()
    {
      set_short_type(fourcc("stsd"));
    }

    void derive_box_version() override;

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    uint32_t m_entry_count = 1;
  };

  class Box_hvc1 : public Box
  {
  public:
    Box_hvc1()
    {
      set_short_type(fourcc("hvc1"));
    }

    void set_hvc1_data(uint16_t img_src_width, uint16_t img_src_height, uint16_t frame_count);

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    uint8_t m_reserved0[6] = {0,};
    uint16_t m_data_reference_index = 1;
    uint16_t m_pre_defined0 = 0;
    uint16_t m_reserved1 = 0;
    uint32_t m_pre_defined1[3] = {0,};
    uint16_t m_width;
    uint16_t m_height;
    uint32_t m_horizresolution = 0x00480000;
    uint32_t m_vertresolution = 0x00480000;
    uint32_t m_reserved2 = 0;
    uint16_t m_frame_count = 1;
    std::string m_compressorname; //= "\013HEVC Coding";
    uint16_t m_depth = 0x0018;
    uint16_t m_pre_defined2 = (uint16_t)0xffff;
  };

  class Box_ccst : public FullBox
  {
  public:
    // typedef union 
    // {
    //   struct {
    //     unsigned m_all_ref_pics_intra : 1;
		// 	  unsigned intra_pred_used : 1;
		// 	  unsigned max_ref_per_pic : 4;
		// 	  unsigned rsvd : 2;
		//   }bits;
		//   uint8_t val;
	  // }ccst_data;

    Box_ccst()
    {
      set_short_type(fourcc("ccst"));
    }

    void derive_box_version() override;

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    uint8_t m_ccst_data = 0;
    uint8_t m_reserved[3] = {0,};
  };

  class Box_stsz : public FullBox
  {
  public:
    Box_stsz()
    {
      set_short_type(fourcc("stsz"));
    }

    void derive_box_version() override;

    void add_entry_size(uint32_t entry_size)
    {
      entry_sizes.push_back(entry_size);
    }

    uint32_t get_sample_offset(uint32_t ID);

    uint32_t get_sample_size(uint32_t ID);

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    uint32_t m_sample_size = 0;
    std::vector<uint32_t> entry_sizes;
  };

  class Box_stts : public FullBox
  {
  public:

    struct sample_info {
      uint32_t m_sample_count;
      uint32_t m_sample_delta;
    };

    Box_stts()
    {
      set_short_type(fourcc("stts"));
    }

    void set_frame_durationinTimeScale(uint64_t duration){
      frame_durationinTimeScales.push_back(duration);
    }

    void derive_box_version() override;

    void calc_stts_data();

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    uint32_t m_entry_count = 0;
    std::vector<sample_info> m_samples;
    std::vector<uint64_t> frame_durationinTimeScales;
  };

  class Box_stsc : public FullBox
  {
  public:
    Box_stsc()
    {
      set_short_type(fourcc("stsc"));
    }

    void add_chunk_sample(){
      m_samples_per_chunk++;
    }

    void derive_box_version() override;

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

    uint32_t get_entry_count() {return m_entry_count;}

    struct sampleofchunk get_chunk_samples(int i) {return m_chunk_samples[i];};

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    uint32_t m_entry_count = 1;
    std::vector<sampleofchunk> m_chunk_samples;
    uint32_t m_first_chunk = 1;
    uint32_t m_samples_per_chunk = 0;
    uint32_t m_sample_description_index = 1;
  };

  class Box_stco : public FullBox
  {
  public:
    Box_stco()
    {
      set_short_type(fourcc("stco"));
    }

    void derive_box_version() override;

    uint32_t get_base_offset() {return m_stco_mdat_offset;}

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

    void patch_iloc_header(StreamWriter& writer, uint64_t mdate_offset);

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    mutable size_t m_stco_box_start = 0;
    uint32_t m_entry_count = 1;
    uint32_t m_stco_mdat_offset = 0;
  };

  class Box_stss : public FullBox
  {
  public:
    Box_stss()
    {
      set_short_type(fourcc("stss"));
    }

    void derive_box_version() override;

    std::string dump(Indent&) const override;

    Error write(StreamWriter& writer) const override;

    void record_sync_data(uint8_t nal_type);

    void set_stss_data();

  protected:
    Error parse(BitstreamRange& range) override;

  private:
    std::vector<bool> m_sync_flags;
    uint32_t m_entry_count = 0;
    std::vector<uint32_t> m_sample_number;
  };

class Box_hdlr : public FullBox
{
public:
  Box_hdlr()
  {
    set_short_type(fourcc("hdlr"));
  }

  std::string dump(Indent&) const override;

  uint32_t get_handler_type() const { return m_handler_type; }

  void set_handler_type(uint32_t handler) { m_handler_type = handler; }

  Error write(StreamWriter& writer) const override;

  void set_name(std::string name) { m_name = std::move(name); }

protected:
  Error parse(BitstreamRange& range) override;

private:
  uint32_t m_pre_defined = 0;
  uint32_t m_handler_type = fourcc("pict");
  uint32_t m_reserved[3] = {0,};
  std::string m_name;
};


class Box_pitm : public FullBox
{
public:
  Box_pitm()
  {
    set_short_type(fourcc("pitm"));
  }

  std::string dump(Indent&) const override;

  heif_item_id get_item_ID() const { return m_item_ID; }

  void set_item_ID(heif_item_id id) { m_item_ID = id; }

  void derive_box_version() override;

  Error write(StreamWriter& writer) const override;

protected:
  Error parse(BitstreamRange& range) override;

private:
  heif_item_id m_item_ID = 0;
};


class Box_iloc : public FullBox
{
public:
  Box_iloc()
  {
    set_short_type(fourcc("iloc"));
  }

  std::string dump(Indent&) const override;

  struct Extent
  {
    uint64_t index = 0;
    uint64_t offset = 0;
    uint64_t length = 0;

    std::vector<uint8_t> data; // only used when writing data
  };

  struct Item
  {
    heif_item_id item_ID = 0;
    uint8_t construction_method = 0; // >= version 1
    uint16_t data_reference_index = 0;
    uint64_t base_offset = 0;

    std::vector<Extent> extents;
  };

  const std::vector<Item>& get_items() const { return m_items; }

  Error read_data(const Item& item,
                  const std::shared_ptr<StreamReader>& istr,
                  const std::shared_ptr<class Box_idat>&,
                  std::vector<uint8_t>* dest) const;

  void set_min_version(uint8_t min_version) { m_user_defined_min_version = min_version; }

  // append bitstream data that will be written later (after iloc box)
  Error append_data(heif_item_id item_ID,
                    const std::vector<uint8_t>& data,
                    uint8_t construction_method = 0);

  // append bitstream data that already has been written (before iloc box)
  // Error write_mdat_before_iloc(heif_image_id item_ID,
  //                              std::vector<uint8_t>& data)

  // reserve data entry that will be written later
  // Error reserve_mdat_item(heif_image_id item_ID,
  //                         uint8_t construction_method,
  //                         uint32_t* slot_ID);
  // void patch_mdat_slot(uint32_t slot_ID, size_t start, size_t length);

  void derive_box_version() override;

  Error write(StreamWriter& writer) const override;

  Error write_mdat_after_iloc(StreamWriter& writer);

    void set_moov_flag(bool flag) {moov_flag = flag;}

protected:
  Error parse(BitstreamRange& range) override;

private:

    bool moov_flag = false;

  std::vector<Item> m_items;

  mutable size_t m_iloc_box_start = 0;
  uint8_t m_user_defined_min_version = 0;
  uint8_t m_offset_size = 0;
  uint8_t m_length_size = 0;
  uint8_t m_base_offset_size = 0;
  uint8_t m_index_size = 0;

  void patch_iloc_header(StreamWriter& writer) const;

  int m_idat_offset = 0; // only for writing: offset of next data array
};


class Box_infe : public FullBox
{
public:
  Box_infe()
  {
    set_short_type(fourcc("infe"));
  }

  std::string dump(Indent&) const override;

  bool is_hidden_item() const { return m_hidden_item; }

  void set_hidden_item(bool hidden);

  heif_item_id get_item_ID() const { return m_item_ID; }

  void set_item_ID(heif_item_id id) { m_item_ID = id; }

  const std::string& get_item_type() const { return m_item_type; }

  void set_item_type(const std::string& type) { m_item_type = type; }

  void set_item_name(const std::string& name) { m_item_name = name; }

  const std::string& get_item_name() const { return m_item_name; }

  const std::string& get_content_type() const { return m_content_type; }

  const std::string& get_content_encoding() const { return m_content_encoding; }

  void set_content_type(const std::string& content_type) { m_content_type = content_type; }

  void set_content_encoding(const std::string& content_encoding) { m_content_encoding = content_encoding; }

  void derive_box_version() override;

  Error write(StreamWriter& writer) const override;

  const std::string& get_item_uri_type() const { return m_item_uri_type; }

  void set_item_uri_type(const std::string& uritype) { m_item_uri_type = uritype; }

protected:
  Error parse(BitstreamRange& range) override;

private:
  heif_item_id m_item_ID = 0;
  uint16_t m_item_protection_index = 0;

  std::string m_item_type;
  std::string m_item_name;
  std::string m_content_type;
  std::string m_content_encoding;
  std::string m_item_uri_type;

  // if set, this item should not be part of the presentation (i.e. hidden)
  bool m_hidden_item = false;
};


class Box_iinf : public FullBox
{
public:
  Box_iinf()
  {
    set_short_type(fourcc("iinf"));
  }

  std::string dump(Indent&) const override;

  void derive_box_version() override;

  Error write(StreamWriter& writer) const override;

protected:
  Error parse(BitstreamRange& range) override;

private:
  //std::vector< std::shared_ptr<Box_infe> > m_iteminfos;
};


class Box_iprp : public Box
{
public:
  Box_iprp()
  {
    set_short_type(fourcc("iprp"));
  }

  std::string dump(Indent&) const override;

protected:
  Error parse(BitstreamRange& range) override;
};


class Box_ipco : public Box
{
public:
  Box_ipco()
  {
    set_short_type(fourcc("ipco"));
  }

  int find_or_append_child_box(const std::shared_ptr<Box>& box);

  Error get_properties_for_item_ID(heif_item_id itemID,
                                   const std::shared_ptr<class Box_ipma>&,
                                   std::vector<std::shared_ptr<Box>>& out_properties) const;

  std::shared_ptr<Box> get_property_for_item_ID(heif_item_id itemID,
                                                const std::shared_ptr<class Box_ipma>&,
                                                uint32_t property_box_type) const;

  bool is_property_essential_for_item(heif_item_id itemId,
                                      const std::shared_ptr<const class Box>& property,
                                      const std::shared_ptr<class Box_ipma>&) const;

  std::string dump(Indent&) const override;

protected:
  Error parse(BitstreamRange& range) override;
};


class Box_ispe : public FullBox
{
public:
  Box_ispe()
  {
    set_short_type(fourcc("ispe"));
  }

  uint32_t get_width() const { return m_image_width; }

  uint32_t get_height() const { return m_image_height; }

  void set_size(uint32_t width, uint32_t height)
  {
    m_image_width = width;
    m_image_height = height;
  }

  std::string dump(Indent&) const override;

  Error write(StreamWriter& writer) const override;

  bool operator==(const Box& other) const override;

protected:
  Error parse(BitstreamRange& range) override;

private:
  uint32_t m_image_width = 0;
  uint32_t m_image_height = 0;
};


class Box_ipma : public FullBox
{
public:
  Box_ipma()
  {
    set_short_type(fourcc("ipma"));
  }

  std::string dump(Indent&) const override;

  struct PropertyAssociation
  {
    bool essential;
    uint16_t property_index;
  };

  const std::vector<PropertyAssociation>* get_properties_for_item_ID(heif_item_id itemID) const;

  bool is_property_essential_for_item(heif_item_id itemId, int propertyIndex) const;

  void add_property_for_item_ID(heif_item_id itemID,
                                PropertyAssociation assoc);

  void derive_box_version() override;

  Error write(StreamWriter& writer) const override;

  void insert_entries_from_other_ipma_box(const Box_ipma& b);

protected:
  Error parse(BitstreamRange& range) override;

  struct Entry
  {
    heif_item_id item_ID;
    std::vector<PropertyAssociation> associations;
  };

  std::vector<Entry> m_entries;
};


class Box_auxC : public FullBox
{
public:
  Box_auxC()
  {
    set_short_type(fourcc("auxC"));
  }

  const std::string& get_aux_type() const { return m_aux_type; }

  void set_aux_type(const std::string& type) { m_aux_type = type; }

  const std::vector<uint8_t>& get_subtypes() const { return m_aux_subtypes; }

  std::string dump(Indent&) const override;

protected:
  Error parse(BitstreamRange& range) override;

  Error write(StreamWriter& writer) const override;

private:
  std::string m_aux_type;
  std::vector<uint8_t> m_aux_subtypes;
};


class Box_irot : public Box
{
public:
  Box_irot()
  {
    set_short_type(fourcc("irot"));
  }

  std::string dump(Indent&) const override;

  int get_rotation() const { return m_rotation; }

  // Only these multiples of 90 are allowed: 0, 90, 180, 270.
  void set_rotation_ccw(int rot) { m_rotation = rot; }

protected:
  Error parse(BitstreamRange& range) override;

  Error write(StreamWriter& writer) const override;

private:
  int m_rotation = 0; // in degrees (CCW)
};


class Box_imir : public Box
{
public:
  Box_imir()
  {
    set_short_type(fourcc("imir"));
  }

  heif_transform_mirror_direction get_mirror_direction() const { return m_axis; }

  void set_mirror_direction(heif_transform_mirror_direction dir) { m_axis = dir; }

  std::string dump(Indent&) const override;

protected:
  Error parse(BitstreamRange& range) override;

  Error write(StreamWriter& writer) const override;

private:
  heif_transform_mirror_direction m_axis = heif_transform_mirror_direction_vertical;
};


class Box_clap : public Box
{
public:
  Box_clap()
  {
    set_short_type(fourcc("clap"));
  }

  std::string dump(Indent&) const override;

  int left_rounded(int image_width) const;  // first column
  int right_rounded(int image_width) const; // last column that is part of the cropped image
  int top_rounded(int image_height) const;   // first row
  int bottom_rounded(int image_height) const; // last row included in the cropped image

  double left(int image_width) const;
  double top(int image_height) const;

  int get_width_rounded() const;

  int get_height_rounded() const;

  void set(uint32_t clap_width, uint32_t clap_height,
           uint32_t image_width, uint32_t image_height);

protected:
  Error parse(BitstreamRange& range) override;

  Error write(StreamWriter& writer) const override;

private:
  Fraction m_clean_aperture_width;
  Fraction m_clean_aperture_height;
  Fraction m_horizontal_offset;
  Fraction m_vertical_offset;
};


class Box_iref : public FullBox
{
public:
  Box_iref()
  {
    set_short_type(fourcc("iref"));
  }

  struct Reference
  {
    BoxHeader header;

    heif_item_id from_item_ID;
    std::vector<heif_item_id> to_item_ID;
  };


  std::string dump(Indent&) const override;

  bool has_references(heif_item_id itemID) const;

  std::vector<heif_item_id> get_references(heif_item_id itemID, uint32_t ref_type) const;

  std::vector<Reference> get_references_from(heif_item_id itemID) const;

  void add_references(heif_item_id from_id, uint32_t type, const std::vector<heif_item_id>& to_ids);

protected:
  Error parse(BitstreamRange& range) override;

  Error write(StreamWriter& writer) const override;

  void derive_box_version() override;

private:
  std::vector<Reference> m_references;
};


//   class Box_hvcC : public Box
//   {
//   public:
//     Box_hvcC()
//     {
//       set_short_type(fourcc("hvcC"));
//       set_is_full_box(false);
//     }

//     Box_hvcC(const BoxHeader& hdr) : Box(hdr)
//     {}

//     struct configuration
//     {
//       uint8_t configuration_version;
//       uint8_t general_profile_space;
//       bool general_tier_flag;
//       uint8_t general_profile_idc;
//       uint32_t general_profile_compatibility_flags;

//       static const int NUM_CONSTRAINT_INDICATOR_FLAGS = 48;
//       std::bitset<NUM_CONSTRAINT_INDICATOR_FLAGS> general_constraint_indicator_flags;

//       uint8_t general_level_idc;

//       uint16_t min_spatial_segmentation_idc;
//       uint8_t parallelism_type;
//       uint8_t chroma_format;
//       uint8_t bit_depth_luma;
//       uint8_t bit_depth_chroma;
//       uint16_t avg_frame_rate;

//       uint8_t constant_frame_rate;
//       uint8_t num_temporal_layers;
//       uint8_t temporal_id_nested;
//     };


//     std::string dump(Indent&) const override;

//     bool get_headers(std::vector<uint8_t>* dest) const;

//     void set_configuration(const configuration& config)
//     { m_configuration = config; }

//     const configuration& get_configuration() const
//     { return m_configuration; }

//     void append_nal_data(const std::vector<uint8_t>& nal);

//     void append_nal_data(const uint8_t* data, size_t size);

//     void append_nal_data_for_movie(const uint8_t* data, size_t size);
//     bool get_header(uint32_t id, std::vector<uint8_t>* dest) const;

//     Error write(StreamWriter& writer) const override;

//   protected:
//     Error parse(BitstreamRange& range) override;

//   private:
//     struct NalArray
//     {
//       uint8_t m_array_completeness;
//       uint8_t m_NAL_unit_type;

//       std::vector<std::vector<uint8_t> > m_nal_units;
//     };

//     configuration m_configuration;
//     uint8_t m_length_size = 4; // default: 4 bytes for NAL unit lengths

//     std::vector<NalArray> m_nal_array;
//   };


//   class Box_av1C : public Box
//   {
//   public:
//     Box_av1C()
//     {
//       set_short_type(fourcc("av1C"));
//       set_is_full_box(false);
//     }

//     Box_av1C(const BoxHeader& hdr) : Box(hdr)
//     {}

//     struct configuration
//     {
//       //unsigned int (1) marker = 1;
//       uint8_t version = 1;
//       uint8_t seq_profile = 0;
//       uint8_t seq_level_idx_0 = 0;
//       uint8_t seq_tier_0 = 0;
//       uint8_t high_bitdepth = 0;
//       uint8_t twelve_bit = 0;
//       uint8_t monochrome = 0;
//       uint8_t chroma_subsampling_x = 0;
//       uint8_t chroma_subsampling_y = 0;
//       uint8_t chroma_sample_position = 0;
//       //uint8_t reserved = 0;

//       uint8_t initial_presentation_delay_present = 0;
//       uint8_t initial_presentation_delay_minus_one = 0;

//       //unsigned int (8)[] configOBUs;
//     };


//     std::string dump(Indent&) const override;

//     bool get_headers(std::vector<uint8_t>* dest) const
//     {
//       *dest = m_config_OBUs;
//       return true;
//     }

//     void set_configuration(const configuration& config)
//     { m_configuration = config; }

//     const configuration& get_configuration() const
//     { return m_configuration; }

//     //void append_nal_data(const std::vector<uint8_t>& nal);
//     //void append_nal_data(const uint8_t* data, size_t size);

//     Error write(StreamWriter& writer) const override;

//   protected:
//     Error parse(BitstreamRange& range) override;

//   private:
//     configuration m_configuration;

//     std::vector<uint8_t> m_config_OBUs;
//   };


class Box_idat : public Box
{
public:
  std::string dump(Indent&) const override;

  Error read_data(const std::shared_ptr<StreamReader>& istr,
                  uint64_t start, uint64_t length,
                  std::vector<uint8_t>& out_data) const;

  int append_data(const std::vector<uint8_t>& data)
  {
    auto pos = m_data_for_writing.size();

    m_data_for_writing.insert(m_data_for_writing.end(),
                              data.begin(),
                              data.end());

    return (int) pos;
  }

  Error write(StreamWriter& writer) const override;

protected:
  Error parse(BitstreamRange& range) override;

  std::streampos m_data_start_pos;

  std::vector<uint8_t> m_data_for_writing;
};


class Box_grpl : public Box
{
public:
  std::string dump(Indent&) const override;

protected:
  Error parse(BitstreamRange& range) override;
};


class Box_EntityToGroup : public FullBox
{
public:
  std::string dump(Indent&) const override;

protected:
  uint32_t group_id;
  std::vector<heif_item_id> entity_ids;

  Error parse(BitstreamRange& range) override;
};


class Box_ster : public Box_EntityToGroup
{
public:
  std::string dump(Indent&) const override;

  heif_item_id get_left_image() const { return entity_ids[0]; }
  heif_item_id get_right_image() const { return entity_ids[1]; }

protected:

  Error parse(BitstreamRange& range) override;
};


class Box_pymd : public Box_EntityToGroup
{
public:
  std::string dump(Indent&) const override;

protected:
  uint16_t tile_size_x;
  uint16_t tile_size_y;

  struct LayerInfo {
    uint16_t layer_binning;
    uint16_t tiles_in_layer_row_minus1;
    uint16_t tiles_in_layer_column_minus1;
  };

  std::vector<LayerInfo> m_layer_infos;

  Error parse(BitstreamRange& range) override;
};




class Box_dinf : public Box
{
public:
    Box_dinf()
    {
      set_short_type(fourcc("dinf"));
    }

  std::string dump(Indent&) const override;

protected:
  Error parse(BitstreamRange& range) override;
};


class Box_dref : public FullBox
{
public:

    Box_dref()
    {
      set_short_type(fourcc("dref"));
    }

    Error write(StreamWriter& writer) const override;

  std::string dump(Indent&) const override;

protected:
  Error parse(BitstreamRange& range) override;
};


class Box_url : public FullBox
{
public:
    Box_url()
    {
      set_short_type(fourcc("url "));
    }

    void derive_box_version() override;

  std::string dump(Indent&) const override;

protected:
  Error parse(BitstreamRange& range) override;

  std::string m_location;
};

class Box_pixi : public FullBox
{
public:
  Box_pixi()
  {
    set_short_type(fourcc("pixi"));
  }

  int get_num_channels() const { return (int) m_bits_per_channel.size(); }

  int get_bits_per_channel(int channel) const { return m_bits_per_channel[channel]; }

  void add_channel_bits(uint8_t c)
  {
    m_bits_per_channel.push_back(c);
  }

  std::string dump(Indent&) const override;

  Error write(StreamWriter& writer) const override;

protected:
  Error parse(BitstreamRange& range) override;

private:
  std::vector<uint8_t> m_bits_per_channel;
};


class Box_pasp : public Box
{
public:
  Box_pasp()
  {
    set_short_type(fourcc("pasp"));
  }

  uint32_t hSpacing = 1;
  uint32_t vSpacing = 1;

  std::string dump(Indent&) const override;

  Error write(StreamWriter& writer) const override;

protected:
  Error parse(BitstreamRange& range) override;
};


class Box_lsel : public Box
{
public:
  Box_lsel()
  {
    set_short_type(fourcc("lsel"));
  }

  uint16_t layer_id = 0;

  std::string dump(Indent&) const override;

  Error write(StreamWriter& writer) const override;

protected:
  Error parse(BitstreamRange& range) override;
};


class Box_clli : public Box
{
public:
  Box_clli()
  {
    set_short_type(fourcc("clli"));

    clli.max_content_light_level = 0;
    clli.max_pic_average_light_level = 0;
  }

  heif_content_light_level clli;

  std::string dump(Indent&) const override;

  Error write(StreamWriter& writer) const override;

protected:
  Error parse(BitstreamRange& range) override;
};


class Box_mdcv : public Box
{
public:
  Box_mdcv();

  heif_mastering_display_colour_volume mdcv;

  std::string dump(Indent&) const override;

  Error write(StreamWriter& writer) const override;

protected:
  Error parse(BitstreamRange& range) override;
};


class Box_cmin : public FullBox
{
public:
  Box_cmin()
  {
    set_short_type(fourcc("cmin"));
  }

  struct AbsoluteIntrinsicMatrix;

  struct RelativeIntrinsicMatrix
  {
    double focal_length_x = 0;
    double principal_point_x = 0;
    double principal_point_y = 0;

    bool is_anisotropic = false;
    double focal_length_y = 0;
    double skew = 0;

    void compute_focal_length(int image_width, int image_height,
                              double& out_focal_length_x, double& out_focal_length_y) const;

    void compute_principal_point(int image_width, int image_height,
                                 double& out_principal_point_x, double& out_principal_point_y) const;

    struct AbsoluteIntrinsicMatrix to_absolute(int image_width, int image_height) const;
  };

  struct AbsoluteIntrinsicMatrix
  {
    double focal_length_x;
    double focal_length_y;
    double principal_point_x;
    double principal_point_y;
    double skew = 0;

    void apply_clap(const Box_clap* clap, int image_width, int image_height) {
      principal_point_x -= clap->left(image_width);
      principal_point_y -= clap->top(image_height);
    }

    void apply_imir(const Box_imir* imir, int image_width, int image_height) {
      switch (imir->get_mirror_direction()) {
        case heif_transform_mirror_direction_horizontal:
          focal_length_x *= -1;
          skew *= -1;
          principal_point_x = image_width - 1 - principal_point_x;
          break;
        case heif_transform_mirror_direction_vertical:
          focal_length_y *= -1;
          principal_point_y = image_height - 1 - principal_point_y;
          break;
        case heif_transform_mirror_direction_invalid:
          break;
      }
    }
  };

  std::string dump(Indent&) const override;

  RelativeIntrinsicMatrix get_intrinsic_matrix() const { return m_matrix; }

  void set_intrinsic_matrix(RelativeIntrinsicMatrix matrix);

protected:
  Error parse(BitstreamRange& range) override;

  Error write(StreamWriter& writer) const override;

private:
  RelativeIntrinsicMatrix m_matrix;

  uint32_t m_denominatorShift = 0;
  uint32_t m_skewDenominatorShift = 0;
};


class Box_cmex : public FullBox
{
public:
  Box_cmex()
  {
    set_short_type(fourcc("cmex"));
  }

  struct ExtrinsicMatrix
  {
    // in micrometers (um)
    int32_t pos_x = 0;
    int32_t pos_y = 0;
    int32_t pos_z = 0;

    bool rotation_as_quaternions = true;
    bool orientation_is_32bit = false;

    double quaternion_x = 0;
    double quaternion_y = 0;
    double quaternion_z = 0;
    double quaternion_w = 1.0;

    // rotation angle in degrees
    double rotation_yaw = 0;   //  [-180 ; 180)
    double rotation_pitch = 0; //  [-90 ; 90]
    double rotation_roll = 0;  //  [-180 ; 180)

    uint32_t world_coordinate_system_id = 0;

    // Returns rotation matrix in row-major order.
    std::array<double,9> calculate_rotation_matrix() const;
  };

  std::string dump(Indent&) const override;

  ExtrinsicMatrix get_extrinsic_matrix() const { return m_matrix; }

  Error set_extrinsic_matrix(ExtrinsicMatrix matrix);

protected:
  Error parse(BitstreamRange& range) override;

  Error write(StreamWriter& writer) const override;

private:
  ExtrinsicMatrix m_matrix;

  bool m_has_pos_x = false;
  bool m_has_pos_y = false;
  bool m_has_pos_z = false;
  bool m_has_orientation = false;
  bool m_has_world_coordinate_system_id = false;

  enum Flags
  {
    pos_x_present = 0x01,
    pos_y_present = 0x02,
    pos_z_present = 0x04,
    orientation_present = 0x08,
    rot_large_field_size = 0x10,
    id_present = 0x20
  };
};




/**
 * User Description property.
 *
 * Permits the association of items or entity groups with a user-defined name, description and tags;
 * there may be multiple udes properties, each with a different language code.
 *
 * See ISO/IEC 23008-12:2022(E) Section 6.5.20.
 */
class Box_udes : public FullBox
{
public:
  Box_udes()
  {
    set_short_type(fourcc("udes"));
  }

  std::string dump(Indent&) const override;

  Error write(StreamWriter& writer) const override;

  /**
   * Language tag.
   *
   * An RFC 5646 compliant language identifier for the language of the text contained in the other properties.
   * Examples: "en-AU", "de-DE", or "zh-CN“.
   * When is empty, the language is unknown or not undefined.
   */
  std::string get_lang() const { return m_lang; }

  /**
   * Set the language tag.
   *
   * An RFC 5646 compliant language identifier for the language of the text contained in the other properties.
   * Examples: "en-AU", "de-DE", or "zh-CN“.
   */
  void set_lang(const std::string lang) { m_lang = lang; }

  /**
   * Name.
   *
   * Human readable name for the item or group being described.
   * May be empty, indicating no name is applicable.
   */
  std::string get_name() const { return m_name; }

  /**
  * Set the name.
  *
  * Human readable name for the item or group being described.
  */
  void set_name(const std::string name) { m_name = name; }

  /**
   * Description.
   *
   * Human readable description for the item or group.
   * May be empty, indicating no description has been provided.
   */
  std::string get_description() const { return m_description; }

  /**
   * Set the description.
   *
   * Human readable description for the item or group.
   */
  void set_description(const std::string description) { m_description = description; }

  /**
   * Tags.
   *
   * Comma separated user defined tags applicable to the item or group.
   * May be empty, indicating no tags have been assigned.
   */
  std::string get_tags() const { return m_tags; }

  /**
   * Set the tags.
   *
   * Comma separated user defined tags applicable to the item or group.
   */
  void set_tags(const std::string tags) { m_tags = tags; }

protected:
  Error parse(BitstreamRange& range) override;

private:
  std::string m_lang;
  std::string m_name;
  std::string m_description;
  std::string m_tags;
};

#endif
