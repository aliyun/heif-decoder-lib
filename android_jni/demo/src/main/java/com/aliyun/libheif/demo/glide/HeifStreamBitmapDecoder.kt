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
import java.io.InputStream

class HeifStreamBitmapDecoder(private val bitmapPool: BitmapPool) : ResourceDecoder<InputStream, Bitmap> {

    override fun handles(source: InputStream, options: Options): Boolean {
        val byteBuffer = ByteBufferUtil.fromStream(source)
        var buffer = ByteBufferUtil.toBytes(byteBuffer)
        return HeifNative.isHeic(buffer.size.toLong(), buffer)
    }

    override fun decode(
        source: InputStream,
        width: Int,
        height: Int,
        options: Options
    ): Resource<Bitmap>? {
        Log.d("StreamBitmapDecoder", "================>heic decode")
        val byteBuffer = ByteBufferUtil.fromStream(source)
        val buffer = ByteBufferUtil.toBytes(byteBuffer)
        var heifInfo = HeifInfo()
        HeifNative.getInfo(heifInfo, buffer.size.toLong(), buffer)
        val heifSize = heifInfo.frameList[0]
        val bitmap = Bitmap.createBitmap(heifSize.width, heifSize.height, Bitmap.Config.ARGB_8888)
        HeifNative.toRgba(buffer.size.toLong(), buffer, bitmap)
        return BitmapResource.obtain(bitmap, bitmapPool)
    }
}