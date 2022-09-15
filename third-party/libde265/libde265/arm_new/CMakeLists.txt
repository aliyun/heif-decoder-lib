set (arm_sources 
	neon_common.cc neon_common.h neon_dbk.cc neon_dbk.h neon_dct.cc neon_dct.h neon_intrapred.cc neon_intrapred.h arm.cc arm.h
)

#set (arm_neon_sources 
#  neon_test.cc neon_test.h
#)

add_library(arm OBJECT ${arm_sources})

#add_library(arm_neon OBJECT ${arm_neon_sources})

#set(ARM_OBJECTS $<TARGET_OBJECTS:arm> $<TARGET_OBJECTS:arm_neon> PARENT_SCOPE)

set(DSP_OBJECTS $<TARGET_OBJECTS:arm>  PARENT_SCOPE)
