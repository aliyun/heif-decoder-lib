package com.aliyun.libheif.demo.glide

import android.content.Context
import android.util.Log
import com.bumptech.glide.Registry
import com.bumptech.glide.annotation.GlideModule
import com.bumptech.glide.module.AppGlideModule

@GlideModule
class HeicGlideModule : AppGlideModule() {

    @Override fun registerComponents(context: Context, registry: Registry) {
        Log.d("HeicGlideModule", "registerComponents")
    }

}