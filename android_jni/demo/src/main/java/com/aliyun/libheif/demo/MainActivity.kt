package com.aliyun.libheif.demo

import android.graphics.Bitmap
import android.os.Bundle
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.aliyun.libheif.HeifInfo
import com.aliyun.libheif.HeifNative
import com.aliyun.libheif.HeifSize
import com.bumptech.glide.Glide
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        decodeImage()
        glideDecodeHeic()
    }


    private fun decodeImage() {
        val image = findViewById<ImageView>(R.id.image);
        val inputStream  = assets.open("test.heic")
        val buffer = ByteArray(8192)
        var bytesRead: Int
        val output = ByteArrayOutputStream()
        while (inputStream.read(buffer).also { bytesRead = it } != -1) {
            output.write(buffer, 0, bytesRead)
        }

        val heifInfo = HeifInfo()
        val fileBuffer: ByteArray = output.toByteArray()

        HeifNative.getInfo(heifInfo, fileBuffer.size.toLong(), fileBuffer, )
        val heifSize = heifInfo.frameList.first()
        val bitmap = Bitmap.createBitmap(
            heifSize.width,
            heifSize.height,
            Bitmap.Config.ARGB_8888)

        val heifBuffer = HeifNative.toRgba(fileBuffer.size.toLong(), fileBuffer, bitmap)
        image.setImageBitmap(bitmap)
    }

    private fun glideDecodeHeic() {
        val imgUrl = "https://oss-console-img-demo-cn-hangzhou.oss-cn-hangzhou.aliyuncs.com/example.jpg?x-oss-process=image/format,heic"
        val glideImg = findViewById<ImageView>(R.id.glideImage)
        Glide.with(this).load(imgUrl).into(glideImg)
    }
}