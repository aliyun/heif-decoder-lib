/*
 * H.265 video codec.
 * ASM speedup module
 * mingyuan.myy@alibaba-inc.com
 */

#ifndef DE265_X86_H
#define DE265_X86_H

#include <iostream>
#include "acceleration.h"

void init_acceleration_functions_sse(struct acceleration_functions* accel);

#endif