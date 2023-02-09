/*
 * Copyright 2019 aliyun
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.aliyun.libheif

class HeifInfo {
    var frameNum = 0
    var duration = 0
    val frameList = ArrayList<HeifSize>()

    @JvmOverloads
    fun addHeifSize(width: Int, height: Int, bitdepth: Int = 8) {
        val newframe = HeifSize()
        newframe.width = width
        newframe.height = height
        newframe.bitDepth = bitdepth
        frameList.add(newframe)
    }
}