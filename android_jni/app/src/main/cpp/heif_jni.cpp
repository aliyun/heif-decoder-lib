
#include <jni.h>
#include "libheif/heif.h"
#include <android/log.h>
#include <cstring>
#include <vector>
#include <android/bitmap.h>

#define TAG "libheif"

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
    }
    heif_context_free(ctx);
    delete [] params.img_params; 
    params.img_params = NULL ;
    return true;
}