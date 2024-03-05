package com.aliyun.libheif.demo.glide

import android.content.Context
import android.graphics.Bitmap
import com.bumptech.glide.Glide
import com.bumptech.glide.Registry
import com.bumptech.glide.annotation.GlideModule
import com.bumptech.glide.module.LibraryGlideModule
import java.io.InputStream
import java.nio.ByteBuffer

@GlideModule(glideName = "MyHeic")
open class MyHeicModule : LibraryGlideModule() {
    override fun registerComponents(context: Context, glide: Glide, registry: Registry) {
        registry.prepend(ByteBuffer::class.java, Bitmap::class.java, HeifByteBufferBitmapDecoder(glide.bitmapPool))
        registry.prepend(InputStream::class.java, Bitmap::class.java, HeifStreamBitmapDecoder(glide.bitmapPool))
    }
}