package com.aliyun.libheif.demo.glide

import android.graphics.Bitmap
import android.util.Log
import com.aliyun.libheif.HeifInfo
import com.aliyun.libheif.HeifNative
import com.bumptech.glide.load.Options
import com.bumptech.glide.load.ResourceDecoder
import com.bumptech.glide.load.engine.Resource
import com.bumptech.glide.load.engine.bitmap_recycle.BitmapPool
import com.bumptech.glide.load.resource.bitmap.BitmapResource
import com.bumptech.glide.util.ByteBufferUtil
import java.nio.ByteBuffer

class HeifByteBufferBitmapDecoder(private val bitmapPool: BitmapPool): ResourceDecoder<ByteBuffer, Bitmap> {
    override fun handles(source: ByteBuffer, options: Options): Boolean {
        val buffer = ByteBufferUtil.toBytes(source)
        return HeifNative.isHeic(buffer.size.toLong(), buffer);
    }

    override fun decode(
        source: ByteBuffer,
        width: Int,
        height: Int,
        options: Options
    ): Resource<Bitmap>? {
        Log.e("ByteBufferBitmapDecoder", "=======================>heic decode")
        val buffer = ByteBufferUtil.toBytes(source)
        var heifInfo = HeifInfo()
        HeifNative.getInfo(heifInfo, buffer.size.toLong(), buffer)
        val heifSize = heifInfo.frameList[0]
        val bitmap = Bitmap.createBitmap(heifSize.width, heifSize.height, Bitmap.Config.ARGB_8888)
        HeifNative.toRgba(buffer.size.toLong(), buffer, bitmap)
        return BitmapResource.obtain(bitmap, bitmapPool)
    }
}