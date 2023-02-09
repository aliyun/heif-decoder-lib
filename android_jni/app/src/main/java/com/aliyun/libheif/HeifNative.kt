package com.aliyun.libheif

import android.graphics.Bitmap


/**
 * how to use :
 *     HeifSize heifSize = new HeifSize();
 *     byte[] buffer = HeifNative.toRgba(heifSize, length, byteArray);
 *     Bitmap bitmap = Bitmap.createBitmap(heifSize.getWidth(), heifSize.getHeight(), Bitmap.Config.ARGB_8888);
 *     bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(buffer));
 */
class HeifNative {

    companion object {
        init {
            try {
                System.loadLibrary("c++_shared")
                System.loadLibrary("heif_jni")
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        /**
         * judge file is heic format or not
         * @param length  the length of effective file memory
         * @param fileBuf file pointer in memory
         * @bitmap 
         * @return if false, decoder images failed
         */
        @JvmStatic
        external fun toRgba(length: Long, fileBuf: ByteArray, bitmap: Bitmap): Boolean


        /**
         * judge file is heic format or not
         * @param index  input, select the index image which to be decoder, start with 0
         * @param length  the length of effective file memory
         * @param fileBuf file pointer in memory
         * @return if false, decoder images failed
         */
        @JvmStatic
        external fun imagesToRgba(index: Int, length: Long, fileBuf: ByteArray, bitmap: Bitmap): Boolean

        /**
         * judge file is heic format or not
         * @param length  the length of effective file memory
         * @param filebuf file pointer in memory
         * @return bool, if true, the format is heif;
         * if false, the fromat is not heif
         */
        @JvmStatic
        external fun isHeic(length: Long, filebuf: ByteArray?): Boolean

        /**
         * get info from heic picture
         * @param outSize output size
         * @param length  the length of effective file memory
         * @param filebuf file pointer in memory
         * @return bool, if true, outSize is valid;
         * if false, outSize is not valid
         */
        @JvmStatic
        external fun getInfo(info: HeifInfo, length: Long, filebuf: ByteArray?): Boolean
    }
}