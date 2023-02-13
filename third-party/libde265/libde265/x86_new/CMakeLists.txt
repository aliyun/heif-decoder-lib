set (x86_sources 
  x86.cc x86.h
)

set (x86_sse_sources 
  x86_intrapred.cc x86_intrapred.h x86_idct.cc x86_idct.h x86_sao.cc x86_sao.h x86_dbk.cc x86_dbk.h
)

add_library(x86 OBJECT ${x86_sources})

add_library(x86_sse OBJECT ${x86_sse_sources})

set(sse_flags "")

if(NOT MSVC)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(sse_flags "${sse_flags} -msse4.1")
  else(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(sse_flags "${sse_flags} -msse2 -mssse3 -msse4.1")
  endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
endif()

set(DSP_OBJECTS $<TARGET_OBJECTS:x86> $<TARGET_OBJECTS:x86_sse> PARENT_SCOPE)

SET_TARGET_PROPERTIES(x86_sse PROPERTIES COMPILE_FLAGS "${sse_flags}")