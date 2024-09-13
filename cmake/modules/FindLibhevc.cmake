include(LibFindMacros)
libfind_pkg_check_modules(LIBHEVC_PKGCONF libhevc)

find_path(LIBHEVC_INCLUDE_DIR
    NAMES decoder/ihevcd_decode.h common/iv.h
    HINTS ${LIBHEVC_PKGCONF_INCLUDE_DIRS} ${LIBHEVC_PKGCONF_INCLUDEDIR}
    PATH_SUFFIXES hevc
)

find_library(LIBHEVC_LIBRARY
    NAMES libhevcdec hevcdec
    HINTS ${LIBHEVC_PKGCONF_LIBRARY_DIRS} ${LIBHEVC_PKGCONF_LIBDIR}
)

set(LIBHEVC_PROCESS_LIBS ${LIBHEVC_LIBRARY})
set(LIBHEVC_PROCESS_INCLUDES ${LIBHEVC_INCLUDE_DIR})
libfind_process(LIBHEVC)

#set(LIBHEVC_PROCESS_LIBS "$LIBHEVC_PROCESS_LIBS/common:$LIBHEVC_PROCESS_LIBS/decoder")

#if(LIBDE265_INCLUDE_DIR)
#  set(libde265_config_file "${LIBDE265_INCLUDE_DIR}/libde265/de265-version.h")
#  if(EXISTS ${libde265_config_file})
#      file(STRINGS
#           ${libde265_config_file}
#           TMP
#           REGEX "#define LIBDE265_VERSION .*$")
#      string(REGEX REPLACE "#define LIBDE265_VERSION" "" TMP ${TMP})
#      string(REGEX MATCHALL "[0-9.]+" LIBDE265_VERSION ${TMP})
#  endif()
#endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBHEVC
    REQUIRED_VARS
    	LIBHEVC_INCLUDE_DIRS
	LIBHEVC_LIBRARIES
)

#set(LIBHEVC_INCLUDE_DIR "${LIBHEVC_INCLUDE_DIRS}/commom" "${LIBHEVC_INCLUDE_DIRS}/decoder")
message(STATUS "yyyy LIBHEVC_INCLUDE_DIR = ${LIBHEVC_INCLUDE_DIR}")
message(STATUS "yyyy LIBHEVC_INCLUDE_DIRS = ${LIBHEVC_INCLUDE_DIRS}")
message(STATUS "yyyy LIBHEVC_LIBRARIES = ${LIBHEVC_LIBRARIES}")
