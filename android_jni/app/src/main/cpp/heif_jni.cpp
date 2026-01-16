
#include <jni.h>
#include "libheif/heif.h"
#include <android/log.h>
#include <cstring>
#include <vector>
#include <android/bitmap.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdlib>

#define TAG "libheif"

enum heif_icc_type {
    icc_none,
    icc_nclx,
    icc_ricc,
    icc_prof 
};

template<typename T>
class JniLocalRef {
public:
    JniLocalRef(JNIEnv* env, T obj) : env_(env), obj_(obj) {}
    ~JniLocalRef() {
        if (obj_) env_->DeleteLocalRef(reinterpret_cast<jobject>(obj_));
    }

    JniLocalRef(const JniLocalRef&) = delete;
    JniLocalRef& operator=(const JniLocalRef&) = delete;

    T get() const { return obj_; }
    bool valid() const { return obj_ != nullptr; }

private:
    JNIEnv* env_;
    T obj_;
};

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_aliyun_libheif_HeifNative_toRgba(JNIEnv *env, jclass type, jlong length, jbyteArray fileBuf, jobject bitmap) {

    jbyte* jfilebuf = env->GetByteArrayElements(fileBuf, NULL);
    char* cfilebuf = (char *)jfilebuf;
    long file_len = length;

    enum heif_filetype_result filetype_check = heif_check_filetype((unsigned char*)cfilebuf, 12);
    if(filetype_check == heif_filetype_no) {
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file error, is not heif or heic format");
        return false ;
    }

    struct heif_error err;
    heif_context* ctx = heif_context_alloc();
    err = heif_context_read_from_memory_without_copy(ctx, cfilebuf, file_len, nullptr);
    if(err.code != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag ctx_read %d ", err.code);
    }

    heif_image_handle* handle = NULL;
    err = heif_context_get_primary_image_handle(ctx, &handle);
    if(err.code != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag handle");
    }
//    heif_context_set_threads(ctx, handle, 2);

    // ================================  bitmap shared memory =============================
    uint8_t strict_decoding = 0;
    struct heif_decoding_options* decode_options = heif_decoding_options_alloc();
    decode_options->strict_decoding = strict_decoding;

    // begin to lock pixel
    AndroidBitmapInfo  bitmap_info;
    int bitmap_info_pass = AndroidBitmap_getInfo(env, bitmap, &bitmap_info);

    unsigned char *resultBitmapPixels = NULL;
    int ret = AndroidBitmap_lockPixels(env, bitmap, (void**)&resultBitmapPixels);
    if (ret < 0 || bitmap_info_pass < 0)
    {
        heif_context_free(ctx);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image %dx%d failed flag bitmap, bitmap_info_pass %d", bitmap_info.width, bitmap_info.height, bitmap_info_pass );
    	return false;
    }

    uint8_t* ext_buf = static_cast<uint8_t*>(resultBitmapPixels);
    uint32_t buf_len = bitmap_info.stride * bitmap_info.height  ;
    heif_decoding_options_add_external_dest(decode_options, ext_buf, buf_len, bitmap_info.stride);

    int bitdepth = heif_image_handle_get_luma_bits_per_pixel(handle);
    int has_alpha = heif_image_handle_has_alpha_channel(handle);
    int is_premul = heif_image_handle_is_premultiplied_alpha(handle);
    __android_log_print(ANDROID_LOG_DEBUG, TAG, "heif_decoding source image:%dx%d bitdepth:%d , bitmap, stride:%d, bitmap_info_pass %d", bitmap_info.width, bitmap_info.height, bitdepth, bitmap_info.stride, bitmap_info_pass );
    struct heif_image* image = NULL;
    err = heif_decode_image(handle, &image, heif_colorspace_RGB, heif_chroma_interleaved_RGBA, decode_options);
    heif_decoding_options_free(decode_options);

    if(err.code != 0) {
        AndroidBitmap_unlockPixels(env, bitmap);
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        heif_image_release(image);
        heif_image_handle_release(handle);
        heif_context_free(ctx);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "decode file failed!");
        return false ;
    }

    if(has_alpha && !is_premul ) {
        err = heif_image_rgba_premultiply_alpha(image);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
    heif_image_release(image);
    heif_image_handle_release(handle);
    heif_context_free(ctx);

    if(err.code != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "heif decode alpha picture premultiply failed!");
        return false ;
    }
    return true ;
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_aliyun_libheif_HeifNative_imagesToRgba(JNIEnv *env, jclass type, jint index, jlong length, jbyteArray fileBuf, jobject bitmap) {

    jbyte* jfilebuf = env->GetByteArrayElements(fileBuf, NULL);
    char* cfilebuf = (char *)jfilebuf;
    long file_len = length;
    int  img_idx  = index ;

    enum heif_filetype_result filetype_check = heif_check_filetype((unsigned char*)cfilebuf, 12);
    if(filetype_check == heif_filetype_no) {
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file error, is not heif or heic format");
        return false ;
    }

    struct heif_error err;
    heif_context* ctx = heif_context_alloc();
    err = heif_context_read_from_memory(ctx, cfilebuf, file_len, nullptr);
    if(err.code != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag ctx_read %d ", err.code);
    }

    int num_images = heif_context_get_number_of_top_level_images(ctx);

    libheif_parameters params;
    params.img_params = NULL ;
    params.img_params = new image_parameters[num_images];

    if(!params.img_params){
        heif_context_free(ctx);
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log parameter malloc error!");
        return false ;
    }

    // get info
    err = heif_context_get_heif_params(ctx, &params);
    if(err.code != 0) {
        heif_context_free(ctx);
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        delete [] params.img_params; 
        params.img_params = NULL ;
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log decode image failed flag ctx_read %d ", err.code);
        return false;
    }

    if(img_idx >= num_images) {
        heif_context_free(ctx);
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        delete [] params.img_params; 
        params.img_params = NULL ;
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log index is bigger than num_images %d ", err.code);
        return false;
    }

    std::vector<heif_item_id> image_IDs(num_images);
    num_images = heif_context_get_list_of_top_level_image_IDs(ctx, image_IDs.data(),num_images);
    heif_image_handle* handle = NULL;
    err = heif_context_get_image_handle(ctx, image_IDs[img_idx], &handle);
    if(err.code != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag handle");
    }
//    heif_context_set_threads(ctx, handle, 2);

    // ================================  bitmap shared memory =============================
    uint8_t strict_decoding = 0;
    struct heif_decoding_options* decode_options = heif_decoding_options_alloc();
    decode_options->strict_decoding = strict_decoding;

    // begin to lock pixel
    AndroidBitmapInfo  bitmap_info;
    int bitmap_info_pass = AndroidBitmap_getInfo(env, bitmap, &bitmap_info);

    unsigned char *resultBitmapPixels = NULL;
    int ret = AndroidBitmap_lockPixels(env, bitmap, (void**)&resultBitmapPixels);
    if (ret < 0 || bitmap_info_pass < 0)
    {
        heif_context_free(ctx);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image %dx%d failed flag bitmap, bitmap_info_pass %d", bitmap_info.width, bitmap_info.height, bitmap_info_pass );
    	return false;
    }

    uint8_t* ext_buf = static_cast<uint8_t*>(resultBitmapPixels);
    uint32_t buf_len = bitmap_info.stride * bitmap_info.height  ;
    heif_decoding_options_add_external_dest(decode_options, ext_buf, buf_len, bitmap_info.stride);

    image_parameters* img_params = NULL;
    img_params = params.img_params + img_idx ;
    __android_log_print(ANDROID_LOG_DEBUG, TAG, "heif decode image sequence, idx %d, width %d, height %d ", img_idx, img_params->img_width, img_params->img_height );
    int has_alpha = heif_image_handle_has_alpha_channel(handle);
    int is_premul = heif_image_handle_is_premultiplied_alpha(handle);
    struct heif_image* image = NULL;
    err = heif_decode_image(handle, &image, heif_colorspace_RGB, heif_chroma_interleaved_RGBA, decode_options);
    heif_decoding_options_free(decode_options);

    if(err.code != 0) {
        AndroidBitmap_unlockPixels(env, bitmap);
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        heif_image_release(image);
        heif_image_handle_release(handle);
        heif_context_free(ctx);
        delete [] params.img_params; 
        params.img_params = NULL ;
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "decode file failed!");
        return 0 ;
    }

    if(has_alpha && !is_premul ) {
        err = heif_image_rgba_premultiply_alpha(image);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
    heif_image_release(image);
    heif_image_handle_release(handle);
    heif_context_free(ctx);
    delete [] params.img_params; 
    params.img_params = NULL ;

    if(err.code != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "heif decode alpha picture premultiply failed!");
        return 0 ;
    }
    return true;
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_aliyun_libheif_HeifNative_isHeic(JNIEnv *env,jclass type, jlong length, jbyteArray fileBuf) {

    jbyte* jfilebuf = env->GetByteArrayElements(fileBuf, NULL);
    char* cfilebuf = (char *)jfilebuf;
    long file_len = length;
    bool result;

    enum heif_filetype_result filetype_check = heif_check_filetype((unsigned char*)cfilebuf, 12);
    if(filetype_check == heif_filetype_no) {
        result  = false;
    }
    else {
        result  = true;
    }

    env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
    return result;
}

// convert nclx to android colorspace
// Mapping between Android ColorSpace.Named and HEIF color profiles:
//
// Android ColorSpace.Named | heif_color_primaries (value)  | heif_transfer_characteristics (value) | Additional Notes
// ------------------------|-------------------------------|-------------------------------------|------------------
// SRGB                   | ..._BT_709_5 (1)              | ..._IEC_61966_2_1 (13)              | Most common combination. sRGB standard.
// LINEAR_SRGB            | ..._BT_709_5 (1)              | ..._linear (8)                       | sRGB color gamut but linear transfer (Gamma 1.0).
// BT709                  | ..._BT_709_5 (1)              | ..._BT_709_5 (1)                    | Rec. 709 standard, used for HDTV.
// DISPLAY_P3             | ..._SMPTE_EG_432_1 (12)       | ..._IEC_61966_2_1 (13)              | Wide gamut for Apple devices. DCI-P3 gamut + sRGB gamma.
// DCI_P3                 | ..._SMPTE_EG_432_1 (12)       | ..._SMPTE_ST_428_1 (17)             | Digital cinema standard. DCI-P3 gamut + 2.6 gamma.
// BT2020                 | ..._BT_2020_2_... (9)         | ..._BT_2020_2_10bit (14) or ..._12bit (15) | UHDTV SDR (standard dynamic range) standard.
// SMPTE_C                | ..._BT_601_6 (6)              | ..._BT_601_6 (6)                    | SDTV (NTSC) standard, common mapping for BT.601.
// NTSC_1953              | ..._BT_470_6_System_M (4)     | ..._BT_470_6_System_M (4)           | Old 1953 NTSC standard, predates SMPTE C.
// (No direct mapping)    | ..._BT_2020_2_... (9)         | ..._BT_2100_0_PQ (16)               | HDR10. Exists in Android but not in Named enum, needs special handling.
// (No direct mapping)    | ..._BT_2020_2_... (9)         | ..._BT_2100_0_HLG (18)              | HDR HLG. Exists in Android but not in Named enum, needs special handling.

static std::string mapNclxToAndroidColorSpaceName(heif_color_primaries colorPrimaries, heif_transfer_characteristics transferChar, heif_matrix_coefficients matrixCoeffs, bool fullRange)
{
    std::string androidColorSpace;

    switch (colorPrimaries) {
        case heif_color_primaries_ITU_R_BT_709_5: // (1) BT.709 / sRGB primaries
            switch (transferChar) {
                case heif_transfer_characteristic_IEC_61966_2_1: // (13) sRGB EOTF
                    androidColorSpace = "SRGB"; // -> ColorSpace.Named.SRGB
                    break;
                case heif_transfer_characteristic_linear: // (8) Linear
                    androidColorSpace = "LINEAR_SRGB"; // -> ColorSpace.Named.LINEAR_SRGB
                    break;
                case heif_transfer_characteristic_ITU_R_BT_709_5: // (1) BT.709 TF
                    androidColorSpace = "BT709"; // -> ColorSpace.Named.BT709
                    break;
                default:
                    // default
                    androidColorSpace = "SRGB";
                    break;
            }
            break;

        case heif_color_primaries_SMPTE_EG_432_1: // (12) DCI-P3 primaries
            if (transferChar == heif_transfer_characteristic_IEC_61966_2_1) { // (13) sRGB EOTF
                androidColorSpace = "DISPLAY_P3"; // -> ColorSpace.Named.DISPLAY_P3
            } else if (transferChar == heif_transfer_characteristic_SMPTE_ST_428_1) { // (17) 2.6 Gamma
                androidColorSpace = "DCI_P3"; // -> ColorSpace.Named.DCI_P3
            } else {
                // default
                androidColorSpace = "DISPLAY_P3";
            }
            break;

        case heif_color_primaries_ITU_R_BT_2020_2_and_2100_0: // (9) BT.2020 / BT.2100 primaries
            switch (transferChar) {
                case heif_transfer_characteristic_ITU_R_BT_2100_0_PQ: // (16) PQ (HDR10)
                    androidColorSpace = "BT2020_PQ"; // -> ColorSpace.get(Named.BT2020_PQ) (API 28+)
                    break;
                case heif_transfer_characteristic_ITU_R_BT_2100_0_HLG: // (18) HLG
                    androidColorSpace = "BT2020_HLG"; // -> ColorSpace.get(Named.BT2020_HLG) (API 30+)
                    break;
                default: // Includes 14, 15 for 10/12-bit SDR
                    androidColorSpace = "BT2020"; // -> ColorSpace.Named.BT2020
                    break;
            }
            break;

        case heif_color_primaries_ITU_R_BT_601_6: // (6) BT.601
        case heif_color_primaries_SMPTE_240M:     // (7) SMPTE 240M
            androidColorSpace = "SMPTE_C"; // -> ColorSpace.Named.SMPTE_C
            break;

        case heif_color_primaries_ITU_R_BT_470_6_System_M: // (4) NTSC
            androidColorSpace = "NTSC_1953"; // -> ColorSpace.Named.NTSC_1953
            break;

        case heif_color_primaries_unspecified: // (2) Unspecified
        default:
            androidColorSpace = "SRGB";
            break;
    }

    return androidColorSpace;
}

static int getIccContent(heif_image_handle* handle, heif_icc_type& iccType, std::string& iccColorSpace) {

    if(!handle) {
        return 0;
    }

    auto colr_profile = heif_image_handle_get_color_profile_type(handle);

    if(colr_profile == heif_color_profile_type_nclx) {
        struct heif_color_profile_nclx* nclx_profile = nullptr;
        struct heif_error err = heif_image_handle_get_nclx_color_profile(handle, &nclx_profile);

        if(err.code == heif_error_Ok && nclx_profile != nullptr) {

            // Map color primaries
            heif_color_primaries colorPrimaries = nclx_profile->color_primaries;
            heif_transfer_characteristics transferChar = nclx_profile->transfer_characteristics;
            heif_matrix_coefficients matrixCoeffs = nclx_profile->matrix_coefficients;
            bool fullRange = nclx_profile->full_range_flag;

            // Map NCLX color profile to Android color space based on color primaries and transfer characteristics
            iccColorSpace = mapNclxToAndroidColorSpaceName(colorPrimaries, transferChar, matrixCoeffs, fullRange);
            iccType = icc_nclx;
            // Clean up
            heif_nclx_color_profile_free(nclx_profile);
        }
    }
    else if (colr_profile == heif_color_profile_type_rICC || colr_profile == heif_color_profile_type_prof) {
        // Handle ICC profile case
        size_t iccSize = heif_image_handle_get_raw_color_profile_size(handle);
        if(iccSize > 0) {
            // For ICC profiles, return the size or some identifying info
            iccType = (colr_profile == heif_color_profile_type_rICC) ? icc_ricc : icc_prof;
        } else {
            iccType = icc_none;
        }
        iccColorSpace = "SRGB";
        return 1;

    } else {
        // No color profile
        iccType = icc_none;
        iccColorSpace = "SRGB";
    }

    return 0;
}

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_aliyun_libheif_HeifNative_getRawIcc(JNIEnv *env, jclass type, jlong length, jbyteArray fileBuf) {

    if(!fileBuf) {
        return nullptr;
    }
    
    jbyte* jfilebuf = env->GetByteArrayElements(fileBuf, NULL);
    if(!jfilebuf) {
        return nullptr;
    }
    uint8_t* cfilebuf = (uint8_t*)jfilebuf;
    long file_len = length;

    enum heif_filetype_result filetype_check = heif_check_filetype(cfilebuf, 12);
    if(filetype_check == heif_filetype_no) {
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file error, is not heif or heic format");
        return nullptr;
    }

    struct heif_error err;
    heif_context* ctx = heif_context_alloc();
    err = heif_context_read_from_memory_without_copy(ctx, cfilebuf, file_len, nullptr);
    if(err.code != 0) {
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        heif_context_free(ctx);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag ctx_read %d ", err.code);
        return nullptr;
    }

    heif_image_handle* handle = NULL;
    err = heif_context_get_primary_image_handle(ctx, &handle);
    if(err.code != 0) {
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        heif_context_free(ctx);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag handle");
        return nullptr;
    }

    // Get ICC profile data
    size_t iccSize = heif_image_handle_get_raw_color_profile_size(handle);
    jbyteArray resultArray = nullptr;
    
    if(iccSize > 0) {
        uint8_t* iccData = new uint8_t[iccSize];
        if(iccData) {
            err = heif_image_handle_get_raw_color_profile(handle, iccData);
            if(err.code == heif_error_Ok) {
                resultArray = env->NewByteArray(iccSize);
                if(resultArray) {
                    env->SetByteArrayRegion(resultArray, 0, iccSize, (const jbyte*)iccData);
                }
            }
            delete [] iccData;
        }
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, TAG, "heif_getInfo iccSize: %zu", iccSize);

    // Clean up
    heif_image_handle_release(handle);
    heif_context_free(ctx);
    env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);

    return resultArray;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_aliyun_libheif_HeifNative_getInfo(JNIEnv *env,jclass type, jobject info, jlong length, jbyteArray fileBuf) {

    jbyte* jfilebuf = env->GetByteArrayElements(fileBuf, NULL);
    char* cfilebuf = (char *)jfilebuf;
    long file_len = length;

    enum heif_filetype_result filetype_check = heif_check_filetype((unsigned char*)cfilebuf, 12);
    if(filetype_check == heif_filetype_no) {
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file error, is not heif or heic format");
        return false;
    }

    heif_context* ctx = heif_context_alloc();

    struct heif_error err;

    err = heif_context_read_from_memory(ctx, cfilebuf, file_len, nullptr);
    if(err.code != 0) {
        heif_context_free(ctx);
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log decode image failed flag ctx_read %d ", err.code);
        return false ;
    }

    int num_images = heif_context_get_number_of_top_level_images(ctx);

    libheif_parameters params;

    params.img_params = NULL ;
    params.img_params = new image_parameters[num_images];

    if(!params.img_params){
        heif_context_free(ctx);
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log parameter malloc error!");
        return false ;
    }

    // get info
    err = heif_context_get_heif_params(ctx, &params);
    if(err.code != 0) {
        heif_context_free(ctx);
        delete [] params.img_params; 
        params.img_params = NULL ;
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log decode image failed flag ctx_read %d ", err.code);
        return false ;
    }

    env->CallVoidMethod(info, env->GetMethodID(env->GetObjectClass(info), "setFrameNum", "(I)V"), params.frame_count);
    env->CallVoidMethod(info, env->GetMethodID(env->GetObjectClass(info), "setDuration", "(I)V"), params.movie_duration);
    for(int idx=0; idx < num_images; idx++) {
        struct image_parameters *img = params.img_params + idx;
        env->CallVoidMethod(info, env->GetMethodID(env->GetObjectClass(info), "addHeifSize", "(II)V"), img->img_width, img->img_height);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "heif_getInfo frames_num: %d, width: %d,height: %d", num_images, img->img_width, img->img_height);
    }

    // get iccProfile
    heif_image_handle* handle = NULL;
    err = heif_context_get_primary_image_handle(ctx, &handle);
    if(err.code != 0) {
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log heif_context_get_primary_image_handle failed flag handle");
    }
    else {

        do 
        {
            heif_icc_type iccType;
            std::string iccColorSpace;
            int result = getIccContent(handle, iccType, iccColorSpace);
        
            // Get the IccType enum class
            JniLocalRef<jclass> iccTypeClassRef(env, env->FindClass("com/aliyun/libheif/IccType"));
            if (!iccTypeClassRef.valid()) {
                __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to find IccType class");
                break;
            }

            // Get the appropriate IccType enum value based on native iccType
            const char* fieldName = nullptr;
            switch (iccType) {
                case icc_nclx:   fieldName = "NCLX"; break;
                case icc_ricc:   fieldName = "RICC"; break;
                case icc_prof:   fieldName = "PROF"; break;
                case icc_none:
                default:         fieldName = "NONE"; break;
            }

            jfieldID iccTypeFieldId = env->GetStaticFieldID(iccTypeClassRef.get(), fieldName, "Lcom/aliyun/libheif/IccType;");
            if (iccTypeFieldId == nullptr) {
                __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to find IccType field ID");
                break;
            }

            JniLocalRef<jobject> iccTypeEnumValueRef(env, env->GetStaticObjectField(iccTypeClassRef.get(), iccTypeFieldId));
            if (!iccTypeEnumValueRef.valid()) {
                __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to get IccType enum value");
                break;
            }

            jfieldID iccTypeField = env->GetFieldID(env->GetObjectClass(info), "iccType", "Lcom/aliyun/libheif/IccType;");
            if (iccTypeField == nullptr) {
                __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to get iccType field ID");
                break;
            }
            env->SetObjectField(info, iccTypeField, iccTypeEnumValueRef.get());
 
            jstring colorSpaceStr = env->NewStringUTF(iccColorSpace.c_str());
            if (!colorSpaceStr) {
                __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to create colorSpace string");
                break;
            }
            JniLocalRef<jstring> colorSpaceRef(env, colorSpaceStr);

            jfieldID colorSpaceField = env->GetFieldID(env->GetObjectClass(info), "colorSpace", "Ljava/lang/String;");
            if (colorSpaceField == nullptr) {
                __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to get colorSpace field ID");
                break;
            }
            env->SetObjectField(info, colorSpaceField, colorSpaceRef.get());

            __android_log_print(ANDROID_LOG_DEBUG, TAG, "heif_getInfo iccType: %d, colorSpace: %s", iccType, iccColorSpace.c_str());
        } while(0);
    }

    // clear up
    heif_image_handle_release(handle);
    heif_context_free(ctx);
    delete [] params.img_params; 
    params.img_params = nullptr ;
    return true;
}