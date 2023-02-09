
#include <jni.h>
#include <libheif/heif.h>
#include <android/log.h>
#include <cstring>
#include <vector>
#include <android/bitmap.h>
//#include <sys/time.h>
//#include <fstream>
//#include <iostream>
//using namespace std;

#define TAG "libheif"

//extern "C"
//JNIEXPORT jint JNICALL
//Java_com_aliyun_libheif_HeifNative_encodeBitmap(JNIEnv *env, jclass type, jbyteArray bytes_,
//                                                     jint width, jint height, jstring outputPath_) {
//    jbyte *bytes = env->GetByteArrayElements(bytes_, NULL);
//    jsize length = env->GetArrayLength(bytes_);
//    const char *outputPath = env->GetStringUTFChars(outputPath_, 0);
//
//    heif_image* image;
//    heif_image_create(width, height, heif_colorspace_RGB, heif_chroma_interleaved_RGBA, &image);
//    heif_image_add_plane(image, heif_channel_interleaved, width, height, 32);
//
//    int stride = 0;
//    uint8_t* p = heif_image_get_plane(image, heif_channel_interleaved, &stride);
//    __android_log_print(ANDROID_LOG_DEBUG, TAG, "stride of image %d, %dx%d, %d", stride, width, height, length);
//
//    std::memcpy(p, bytes, static_cast<size_t>(length));
//
//    heif_context* ctx = heif_context_alloc();
//    heif_encoder* encoder;
//    heif_context_get_encoder_for_format(ctx, heif_compression_HEVC, &encoder);
//    heif_encoder_set_logging_level(encoder, 4);
//
//    heif_encoding_options* encoding_options = heif_encoding_options_alloc();
//    encoding_options->save_alpha_channel = 0; // must be turned off for Android
//
//    heif_error error;
//    heif_image_handle* handle;
//    error = heif_context_encode_image(ctx, image, encoder, encoding_options, &handle);
//    if (error.code != heif_error_Ok) {
//        __android_log_print(ANDROID_LOG_ERROR, TAG, "encode image error");
//    } else {
//        int ow = heif_image_handle_get_width(handle);
//        int oh = heif_image_handle_get_height(handle);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "encode image done %dx%d", ow, oh);
//    }
//    heif_encoder_release(encoder);
//
//    error = heif_context_write_to_file(ctx, outputPath);
//    if (error.code != heif_error_Ok) {
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "write to file failed");
//    } else {
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "write to file success");
//    }
//
//    heif_image_handle_release(handle);
//    heif_image_release(image);
//
//    heif_context_free(ctx);
//
//    env->ReleaseByteArrayElements(bytes_, bytes, 0);
//    env->ReleaseStringUTFChars(outputPath_, outputPath);
//
//    return error.code;
//}
//
//extern "C"
//JNIEXPORT jbyteArray JNICALL
//Java_com_aliyun_libheif_HeifNative_decodeHeif2RGBA(JNIEnv *env, jclass type, jobject outSize,
//                                                        jstring srcPath_) {
//    const char *srcPath = env->GetStringUTFChars(srcPath_, 0);
//
//    struct heif_error err;
//    heif_context* ctx = heif_context_alloc();
//    err = heif_context_read_from_file(ctx, srcPath, nullptr);
//    if(err.code != 0) {
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag ctx_read %d ", err.code);
//    }
//
//    heif_image_handle* handle;
//    err = heif_context_get_primary_image_handle(ctx, &handle);
//    if(err.code != 0) {
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag handle");
//    }
//
//    heif_image* image;
//    heif_decode_image(handle, &image, heif_colorspace_RGB, heif_chroma_interleaved_RGBA, nullptr);
//
//    int width = heif_image_handle_get_width(handle);
//    int height = heif_image_handle_get_height(handle);
//
//    int stride = 0;
//    const uint8_t* data = heif_image_get_plane_readonly(image, heif_channel_interleaved, &stride);
//
//    jbyteArray array = env->NewByteArray(stride*height);
//    env->SetByteArrayRegion(array, 0, stride*height, reinterpret_cast<const jbyte *>(data));
//
//    env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), width);
//    env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), height);
//
//    env->ReleaseStringUTFChars(srcPath_, srcPath);
//
//    heif_image_release(image);
//    heif_image_handle_release(handle);
//    heif_context_free(ctx);
//
////    __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image %dx%d, stride = %d, run time is %d ms, fps=%5.2f ", width, height, stride,int(1000*secs),(1/secs));
//
//    return array;
//}
//
//
//extern "C"
//JNIEXPORT jbyteArray JNICALL
//Java_com_aliyun_libheif_HeifNative_decodeHeif2RGBAMem(JNIEnv *env, jclass type, jobject outSize, jlong length, jbyteArray fileBuf, jstring srcPath_) {
//
//    jbyte* jfilebuf = env->GetByteArrayElements(fileBuf, NULL);
//    char* cfilebuf = (char *)jfilebuf;
//    long file_len = length;
//
////    const char *srcPath = env->GetStringUTFChars(srcPath_, 0);
////
////    std::ifstream InFile(srcPath, std::ios_base::binary);
////    InFile.seekg(0, std::ios::end);
////    long len = InFile.tellg();
////    char* fileBuf = new char[len];
////    InFile.seekg(0, std::ios::beg);
////    InFile.read(fileBuf, len);
////    InFile.close();
////
////    if(len != file_len) {
////        __android_log_print(ANDROID_LOG_DEBUG, TAG, "file length is different, FilePath:%ld , fileBuf:%ld", len, file_len);
////    }
////    int  hit = 0;
////    int  addr= 0;
////    for(long i=0;i<len; i++) {
////        if(*(cfilebuf + i) != *(fileBuf + i)) {
////            hit = 1;
////            addr= i;
////            break;
////        }
////    }
////    if(hit) {
////        __android_log_print(ANDROID_LOG_DEBUG, TAG, " the contecnt is different, start address : %08x", addr);
////        __android_log_print(ANDROID_LOG_DEBUG, TAG, "00000000: left : FilePath                  right : FileBuf");
////        for(long i=0;i<len; i+=8) {
////            __android_log_print(ANDROID_LOG_DEBUG, TAG, "%08X: %02X %02X %02X %02X %02X %02X %02X %02X      %02X %02X %02X %02X %02X %02X %02X %02X ", i, 
////                                *(cfilebuf+i+0),*(cfilebuf+i+1),*(cfilebuf+i+2),*(cfilebuf+i+3),*(cfilebuf+i+4),*(cfilebuf+i+5),*(cfilebuf+i+6),*(cfilebuf+i+7),
////                                *(FileBuf +i+0),*(FileBuf +i+1),*(FileBuf +i+2),*(FileBuf +i+3),*(FileBuf +i+4),*(FileBuf +i+5),*(FileBuf +i+6),*(FileBuf +i+7));
////        }
////    }
////
////    delete [] FileBuf;
//
//
//    enum heif_filetype_result filetype_check = heif_check_filetype((unsigned char*)cfilebuf, 12);
//    if(filetype_check == heif_filetype_no) {
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), 0);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), 0);
//        env->ReleaseByteArrayElements(filebuf, (jbyte*)jfilebuf, 0);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file error, is not heif or heic format");
//        return 0 ;
//    }
//
//    struct heif_error err;
//    heif_context* ctx = heif_context_alloc();
//    err = heif_context_read_from_memory_without_copy(ctx, cfilebuf, file_len, nullptr);
//    if(err.code != 0) {
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag ctx_read %d ", err.code);
//    }
//
//    heif_image_handle* handle;
//    err = heif_context_get_primary_image_handle(ctx, &handle);
//    if(err.code != 0) {
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag handle");
//    }
//
//    heif_image* image;
//    heif_decode_image(handle, &image, heif_colorspace_RGB, heif_chroma_interleaved_RGBA, nullptr);
//
//    int width = heif_image_handle_get_width(handle);
//    int height = heif_image_handle_get_height(handle);
//
//    int stride = 0;
//    const uint8_t* data = heif_image_get_plane_readonly(image, heif_channel_interleaved, &stride);
//
//    jbyteArray array = env->NewByteArray(stride*height);
//    env->SetByteArrayRegion(array, 0, stride*height, reinterpret_cast<const jbyte *>(data));
//
//    env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), width);
//    env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), height);
//
//    env->ReleaseByteArrayElements(filebuf, (jbyte*)jfilebuf, 0);
//    heif_image_release(image);
//    heif_image_handle_release(handle);
//    heif_context_free(ctx);
//
//    return array;
//}
//
//
//extern "C"
//JNIEXPORT jboolean JNICALL
//Java_com_aliyun_libheif_HeifNative_decodeHeif2RGBAbitmap(JNIEnv *env, jobject /*thiz*/, jstring srcPath_, jobject bitmap) {
//    const char *srcPath = env->GetStringUTFChars(srcPath_, 0);
//
////    struct timeval tv_start ;
////    gettimeofday(&tv_start,NULL);
//
//    heif_context* ctx = heif_context_alloc();
//    if(!ctx){
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag ctx");
//        return false ;
//    }
//    struct heif_error err;
//    err = heif_context_read_from_file(ctx, srcPath, nullptr);
//    if(err.code != 0) {
//        heif_context_free(ctx);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag ctx_read %d ", err.code);
//        return false ;
//    }
//
//    heif_image_handle* handle;
//    err = heif_context_get_primary_image_handle(ctx, &handle);
//    if(err.code != 0) {
//        heif_context_free(ctx);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag handle %d ", err.code);
//        return false ;
//    }
//
//    uint8_t strict_decoding = 0;
//    struct heif_decoding_options* decode_options = heif_decoding_options_alloc();
//    decode_options->strict_decoding = strict_decoding;
//
//    // begin to lock pixel
//    AndroidBitmapInfo  bitmap_info;
//    int bitmap_info_pass = AndroidBitmap_getInfo(env, bitmap, &bitmap_info);
//
//    unsigned char *resultBitmapPixels = NULL;
//    int ret = AndroidBitmap_lockPixels(env, bitmap, (void**)&resultBitmapPixels);
//    if (ret < 0 || bitmap_info_pass < 0)
//    {
//        heif_context_free(ctx);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image %dx%d failed flag bitmap, bitmap_info_pass %d", bitmap_info.width, bitmap_info.height, bitmap_info_pass );
//    	return false ;
//    }
//
//    uint8_t* ext_buf = static_cast<uint8_t*>(resultBitmapPixels);
//    uint32_t buf_len = bitmap_info.width * bitmap_info.height * 4 ;
//    heif_decoding_options_add_external_dest(decode_options, ext_buf, buf_len, bitmap_info.stride);
//
//    heif_image* image;
//    err = heif_decode_image(handle, &image, heif_colorspace_RGB, heif_chroma_interleaved_RGBA, decode_options);
//    heif_decoding_options_free(decode_options);
//    if(err.code != 0) {
//        heif_context_free(ctx);
//        heif_image_handle_release(handle);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image failed flag decode %d ", err.code);
//        return false ;
//    }
//
//    AndroidBitmap_unlockPixels(env, bitmap);//最终得到的就是bitmap
//    env->ReleaseStringUTFChars(srcPath_, srcPath);
//
////    // calculate api run time end
////    struct timeval tv_end ;
////    gettimeofday(&tv_end,NULL);
////    double secs = tv_end.tv_sec-tv_start.tv_sec;
////    secs += (tv_end.tv_usec - tv_start.tv_usec)*0.001*0.001;
////    if(secs == 0) secs += 0.001;
//
//    heif_image_release(image);
//    heif_image_handle_release(handle);
//    heif_context_free(ctx);
//
////    __android_log_print(ANDROID_LOG_DEBUG, TAG, "jni log decode image %dx%d, stride = %d, run time is %d ms, fps=%5.2f ", bitmap_info.width, bitmap_info.height, bitmap_info.stride,int(1000*secs),(1/secs));
//    return true ;
//}
//
//extern "C"
//JNIEXPORT jint JNICALL
//Java_com_aliyun_libheif_HeifNative_encodeYUV(JNIEnv *env, jclass type, jbyteArray bytes_,
//                                                  jint width, jint height, jstring outputPath_) {
//    jbyte *bytes = env->GetByteArrayElements(bytes_, NULL);
//    const char *outputPath = env->GetStringUTFChars(outputPath_, 0);
//
//    heif_image* image;
//    heif_image_create(width, height, heif_colorspace_YCbCr, heif_chroma_420, &image);
//    heif_image_add_plane(image, heif_channel_Y, width, height, 8);
//    heif_image_add_plane(image, heif_channel_Cb, width/2, height/2, 8);
//    heif_image_add_plane(image, heif_channel_Cr, width/2, height/2, 8);
//
//    int sy, su, sv;
//    uint8_t* py = heif_image_get_plane(image, heif_channel_Y, &sy);
//    uint8_t* pu = heif_image_get_plane(image, heif_channel_Cb, &su);
//    uint8_t* pv = heif_image_get_plane(image, heif_channel_Cr, &sv);
//
//    std::memcpy(py, bytes, static_cast<size_t>(width * height));
//    std::memcpy(pu, bytes+(width*height), static_cast<size_t>(width * height / 4));
//    std::memcpy(pv, bytes+(width*height+width*height/4), static_cast<size_t>(width * height / 4));
//
//    heif_context* ctx = heif_context_alloc();
//    heif_encoder* encoder;
//    heif_context_get_encoder_for_format(ctx, heif_compression_HEVC, &encoder);
//
//    heif_encoding_options* options = heif_encoding_options_alloc();
//    options->save_alpha_channel = 0;
//
//    heif_image_handle* handle;
//    heif_context_encode_image(ctx, image, encoder, options, &handle);
//    heif_encoder_release(encoder);
//
//    heif_error error;
//    error = heif_context_write_to_file(ctx, outputPath);
//    if (error.code != heif_error_Ok) {
//        __android_log_print(ANDROID_LOG_ERROR, TAG, "YUV write to file error %s", error.message);
//    } else {
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "YUV write to file success");
//    }
//
//    heif_image_handle_release(handle);
//    heif_image_release(image);
//    heif_context_free(ctx);
//
//    env->ReleaseByteArrayElements(bytes_, bytes, 0);
//    env->ReleaseStringUTFChars(outputPath_, outputPath);
//
//    return error.code;
//}
//
//
//extern "C"
//JNIEXPORT jboolean JNICALL
//Java_com_aliyun_libheif_HeifNative_isHeicImageAndGetInfo(JNIEnv *env,jclass type, jobject outSize, jstring srcPath_) {
//
//    const char *infoPath = env->GetStringUTFChars(srcPath_, 0);
////    __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log decode image srcPath %s", infoPath);
//
//    std::ifstream istr(infoPath, std::ios_base::binary);
//    unsigned char magic[12];
//    istr.read((char*)magic, 12);
//    enum heif_filetype_result filetype_check = heif_check_filetype(magic, 12);
//    istr.close();
//    if(filetype_check == heif_filetype_no) {
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), 0);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), 0);
//        env->ReleaseStringUTFChars(srcPath_, infoPath);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file error, is not heif or heic format");
//        return false;
//    }
//
//    heif_context* ctx = heif_context_alloc();
//
//    struct heif_error err;
//    err = heif_context_read_from_file(ctx, infoPath, nullptr);
//    if(err.code != 0) {
//        heif_context_free(ctx);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), 0);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), 0);
//        env->ReleaseStringUTFChars(srcPath_, infoPath);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log decode image failed flag ctx_read %d ", err.code);
//        return false ;
//    }
//
//    // get info
//    int width = 0;
//    int height= 0;
//    err = heif_context_get_image_info(ctx, &width, &height);
//    if(err.code != 0) {
//        heif_context_free(ctx);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), 0);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), 0);
//        env->ReleaseStringUTFChars(srcPath_, infoPath);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log decode image failed flag ctx_read %d ", err.code);
//        return false ;
//    }
//
//    env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), width);
//    env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), height);
//    env->ReleaseStringUTFChars(srcPath_, infoPath);
//    heif_context_free(ctx);
////    __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file is heif/heic format width %d, height %d", width, height);
//    return true;
//}

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

    heif_image_handle* handle;
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
    __android_log_print(ANDROID_LOG_DEBUG, TAG, "heif_decoding source image:%dx%d bitdepth:%d , bitmap, stride:%d, bitmap_info_pass %d", bitmap_info.width, bitmap_info.height, bitdepth, bitmap_info.stride, bitmap_info_pass );
    heif_image* image;
    err = heif_decode_image(handle, &image, heif_colorspace_RGB, heif_chroma_interleaved_RGBA, decode_options);
    heif_decoding_options_free(decode_options);
    AndroidBitmap_unlockPixels(env, bitmap);
    if(err.code != 0) {
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        heif_image_release(image);
        heif_image_handle_release(handle);
        heif_context_free(ctx);
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "decode file failed!");
        return false ;
    }

    int SrcWidth = heif_image_handle_get_width(handle);
    int SrcHeight = heif_image_handle_get_height(handle);

//    int stride = 0;
//    const uint8_t* data = heif_image_get_plane_readonly(image, heif_channel_interleaved, &stride);
//    __android_log_print(ANDROID_LOG_DEBUG, TAG, "video SrcWidth %d, SrcHeight %d, stride %d ", SrcWidth, SrcHeight, stride );

    env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
    heif_image_release(image);
    heif_image_handle_release(handle);
    heif_context_free(ctx);

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
    heif_image_handle* handle;
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
    heif_image* image;
    err = heif_decode_image(handle, &image, heif_colorspace_RGB, heif_chroma_interleaved_RGBA, decode_options);
    heif_decoding_options_free(decode_options);
    AndroidBitmap_unlockPixels(env, bitmap);
    if(err.code != 0) {
        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
        heif_image_release(image);
        heif_image_handle_release(handle);
        heif_context_free(ctx);
        delete [] params.img_params; 
        params.img_params = NULL ;
        __android_log_print(ANDROID_LOG_DEBUG, TAG, "decode file failed!");
        return 0 ;
    }

    int SrcWidth = heif_image_handle_get_width(handle);
    int SrcHeight = heif_image_handle_get_height(handle);

//    int stride = 0;
//    const uint8_t* data = heif_image_get_plane_readonly(image, heif_channel_interleaved, &stride);
//    __android_log_print(ANDROID_LOG_DEBUG, TAG, "video SrcWidth %d, SrcHeight %d, stride %d ", SrcWidth, SrcHeight, stride );

    env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
    heif_image_release(image);
    heif_image_handle_release(handle);
    heif_context_free(ctx);
    delete [] params.img_params; 
    params.img_params = NULL ;

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

//extern "C"
//JNIEXPORT jboolean JNICALL
//Java_com_aliyun_libheif_HeifNative_getInfo(JNIEnv *env,jclass type, jobject outSize, jlong length, jbyteArray fileBuf) {
//
//    jbyte* jfilebuf = env->GetByteArrayElements(fileBuf, NULL);
//    char* cfilebuf = (char *)jfilebuf;
//    long file_len = length;
//
//    enum heif_filetype_result filetype_check = heif_check_filetype((unsigned char*)cfilebuf, 12);
//    if(filetype_check == heif_filetype_no) {
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), 0);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), 0);
//        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file error, is not heif or heic format");
//        return false;
//    }
//
//    heif_context* ctx = heif_context_alloc();
//
//    struct heif_error err;
//
//    err = heif_context_read_from_memory(ctx, cfilebuf, file_len, nullptr);
//    if(err.code != 0) {
//        heif_context_free(ctx);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), 0);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), 0);
//        env->ReleaseByteArrayElements(fileBuf, (jbyte*)jfilebuf, 0);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log decode image failed flag ctx_read %d ", err.code);
//        return false ;
//    }
//
//    // get info
//    int width = 0;
//    int height= 0;
//    err = heif_context_get_image_info(ctx, &width, &height);
//    if(err.code != 0) {
//        heif_context_free(ctx);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), 0);
//        env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), 0);
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "info log decode image failed flag ctx_read %d ", err.code);
//        return false ;
//    }
//
//    env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setWidth", "(I)V"), width);
//    env->CallVoidMethod(outSize, env->GetMethodID(env->GetObjectClass(outSize), "setHeight", "(I)V"), height);
//    heif_context_free(ctx);
////    __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file is heif/heic format width %d, height %d", width, height);
//    return true;
//}


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

//    __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file is heif/heic format, total fram num %d, duration %d", params.frame_count, params.movie_duration);
//    for(int cnt = 0 ;cnt < num_images; cnt++) {
//        __android_log_print(ANDROID_LOG_DEBUG, TAG, "======= frame : %d, width %d, height %d", cnt, params.img_params[cnt].img_width, params.img_params[cnt].img_height);
//    }

    env->CallVoidMethod(info, env->GetMethodID(env->GetObjectClass(info), "setFrameNum", "(I)V"), params.frame_count);
    env->CallVoidMethod(info, env->GetMethodID(env->GetObjectClass(info), "setDuration", "(I)V"), params.movie_duration);
    for(int idx=0; idx < num_images; idx++) {
        struct image_parameters *img = params.img_params + idx;
//        env->CallVoidMethod(info, env->GetMethodID(env->GetObjectClass(info), "addHeifSize", "(III)V"), img->img_width, img->img_height, img->img_bitdepth );
        env->CallVoidMethod(info, env->GetMethodID(env->GetObjectClass(info), "addHeifSize", "(II)V"), img->img_width, img->img_height);
    }
    heif_context_free(ctx);
    delete [] params.img_params; 
    params.img_params = NULL ;
//    __android_log_print(ANDROID_LOG_DEBUG, TAG, "Input file is heif/heic format width %d, height %d", width, height);
    return true;
}