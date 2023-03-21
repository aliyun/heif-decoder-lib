noinst_LTLIBRARIES = libde265_x86.la  libde265_x86_sse.la

libde265_x86_la_CXXFLAGS = -I$(top_srcdir)/libde265 $(CFLAG_VISIBILITY)
libde265_x86_la_SOURCES = x86.cc x86.h
libde265_x86_la_LIBADD = libde265_x86_sse.la

if HAVE_VISIBILITY
 libde265_x86_la_CXXFLAGS += -DHAVE_VISIBILITY
endif


# SSE4 specific functions

libde265_x86_sse_la_CXXFLAGS = -I$(top_srcdir) -I$(top_srcdir)/libde265 $(CFLAG_VISIBILITY)
if HAVE_SSE4_1
libde265_x86_sse_la_CXXFLAGS += -msse4.1
endif

if HAVE_AVX2
libde265_x86_sse_la_CXXFLAGS += -mavx2
endif

libde265_x86_sse_la_SOURCES = x86_intrapred.cc x86_intrapred.h x86_idct.cc x86_idct.h x86_sao.cc x86_sao.h x86_dbk.cc x86_dbk.h

if HAVE_VISIBILITY
 libde265_x86_sse_la_CXXFLAGS += -DHAVE_VISIBILITY
endif

EXTRA_DIST = \
  CMakeLists.txt
