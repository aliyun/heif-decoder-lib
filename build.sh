#!/bin/bash

if [ ! -d build ]; then
    mkdir build
fi
cd build
rm -rf ./*

export PKG_CONFIG_PATH=/Users/qianchao.cw/local/libpng/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/Users/qianchao.cw/local/libyuv/lib/pkgconfig:$PKG_CONFIG_PATH
# export PKG_CONFIG_PATH=/usr/local/Cellar/libpng/1.6.40/lib/pkgconfig:$PKG_CONFIG_PATH



cmake ../ -DCMAKE_BUILD_TYPE=Debug \
          -DWITH_X265=ON \
          -DWITH_AOM_ENCODER=ON \
          -DWITH_AOM_DECODER=ON \
          -DWIDTH_LIBDE265=ON \
          -DWITH_SvtEnc=OFF \
          -DWITH_RAV1E=OFF
        #   -DWITH_AOM_DECODER_PLUGIN=OFF \
        #   -DWITH_AOM_ENCODER_PLUGIN=OFF \
make -j