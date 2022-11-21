package com.aliyun.libheif.demo

import android.graphics.Bitmap
import android.os.Bundle
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.aliyun.libheif.HeifNative
import com.aliyun.libheif.HeifSize
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        decodeImage()
    }


    private fun decodeImage() {
        val image = findViewById<ImageView>(R.id.image);
        val heifSize = HeifSize();
        val inputStream  = assets.open("test.heic")
        val buffer = ByteArray(8192)
        var bytesRead: Int
        val output = ByteArrayOutputStream()
        while (inputStream.read(buffer).also { bytesRead = it } != -1) {
            output.write(buffer, 0, bytesRead)
        }

        val fileBuffer: ByteArray = output.toByteArray()
        val heifBuffer = HeifNative.toRgba(heifSize, fileBuffer.size.toLong(), fileBuffer)
        val bitmap = Bitmap.createBitmap(heifSize.width, heifSize.height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(heifBuffer));
        image.setImageBitmap(bitmap)
    }
}