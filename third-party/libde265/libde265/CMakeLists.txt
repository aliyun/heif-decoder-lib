include(CMakePackageConfigHelpers)

set (libde265_sources 
  alloc_pool.cc
  bitstream.cc
  cabac.cc
  configparam.cc
  contextmodel.cc
  de265.cc
  deblock.cc
  decctx.cc
  dpb.cc
  en265.cc
  fallback-dct.cc
  fallback-intrapred.cc
  fallback-motion.cc 
  fallback-postfilter.cc
  fallback.cc
  image-io.cc
  image.cc
  intrapred.cc
  md5.cc
  motion.cc
  nal-parser.cc
  nal.cc
  pps.cc
  quality.cc
  refpic.cc
  sao.cc
  scan.cc
  sei.cc
  slice.cc
  sps.cc
  threads.cc
  transform.cc
  util.cc
  visualize.cc
  vps.cc
  vui.cc
)

set (libde265_headers
  acceleration.h
  alloc_pool.h
  bitstream.h
  cabac.h
  configparam.h
  de265-version.h
  contextmodel.h
  de265.h
  deblock.h
  decctx.h
  dpb.h
  en265.h
  fallback-dct.h
  fallback-intrapred.h
  fallback-motion.h
  fallback-postfilter.h
  fallback.h
  image-io.h
  image.h
  intrapred.h
  md5.h
  motion.h
  nal-parser.h
  nal.h
  pps.h
  quality.h
  refpic.h
  sao.h
  scan.h
  sei.h
  slice.h
  sps.h
  threads.h
  transform.h
  util.h
  visualize.h
  vps.h
  vui.h
)

if(MSVC OR MINGW)
  set (libde265_sources
    ${libde265_sources}
    ../extra/win32cond.c
    ../extra/win32cond.h
  )
endif()

add_definitions(-DLIBDE265_EXPORTS)

add_subdirectory (encoder)

if(NOT DISABLE_ARM)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" cpu_lowercase)
  if(cpu_lowercase MATCHES "^arm")
    set(TARGET_CPU "arm")
    add_definitions(-DHAVE_ARM)
    add_subdirectory (arm_new)
    SET(DISABLE_SSE ON)
  elseif(cpu_lowercase MATCHES "aarch64")
    set(TARGET_CPU "arm64")
    add_definitions(-DHAVE_ARM)
    add_definitions(-DHAVE_ARM64)
    add_subdirectory (arm_new)
    SET(DISABLE_SSE ON)
  else()
    #message(WARNING "The architecture ${CMAKE_SYSTEM_PROCESSOR} is not " 
    #                "supported, falling back to the generic target")
    SET(TARGET_CPU "generic")
  endif()
  #message(WARING "The architecture is ${TARGET_CPU}")
endif()

if(NOT DISABLE_SSE)
  if (MSVC)
    set(SUPPORTS_SSE2 1)
    set(SUPPORTS_SSSE3 1)
    set(SUPPORTS_SSE4_1 1)
  else (MSVC)
    check_c_compiler_flag(-msse2 SUPPORTS_SSE2)
    check_c_compiler_flag(-mssse3 SUPPORTS_SSSE3)
    check_c_compiler_flag(-msse4.1 SUPPORTS_SSE4_1)
    check_c_compiler_flag(-mavx SUPPORTS_AVX)
    check_c_compiler_flag(-mavx2 SUPPORTS_AVX2)
    check_c_compiler_flag(-mavx512f SUPPORTS_AVX512F)
    check_c_compiler_flag(-mavx512bw SUPPORTS_AVX512BW)
  endif (MSVC)

  message(WARNING "SUPPORTS_AVX ${SUPPORTS_AVX}")
  message(WARNING "SUPPORTS_AVX2 ${SUPPORTS_AVX2}")
  message(WARNING "SUPPORTS_AVX512F ${SUPPORTS_AVX512F}")
  message(WARNING "SUPPORTS_AVX512BW ${SUPPORTS_AVX512BW}")

  if(SUPPORTS_SSE4_1)
    add_definitions(-DHAVE_SSE4_1)
  endif()

  if(SUPPORTS_AVX AND SUPPORTS_AVX2)
    add_definitions(-DHAVE_AVX2)
    set(CMAKE_CXX_FLAGS "-mavx")
    set(CMAKE_CXX_FLAGS "-mavx2" )
  endif()

  if(SUPPORTS_AVX512F AND SUPPORTS_AVX512BW)
    add_definitions(-DHAVE_AVX512)
  endif()



  if(SUPPORTS_SSE4_1 OR (SUPPORTS_SSE2 AND SUPPORTS_SSSE3) OR (SUPPORTS_AVX AND SUPPORTS_AVX2) OR (SUPPORTS_AVX512F AND SUPPORTS_AVX512BW))
    add_subdirectory (x86_new)
  endif()
endif()



add_library(${PROJECT_NAME} ${libde265_sources} ${ENCODER_OBJECTS} ${DSP_OBJECTS})
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

write_basic_package_version_file(${PROJECT_NAME}ConfigVersion.cmake COMPATIBILITY ExactVersion)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Config
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${libde265_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/de265-version.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(EXPORT ${PROJECT_NAME}Config DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake DESTINATION
    "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
