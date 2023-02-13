/*
 * H.265 video codec.
 * Copyright (c) 2013-2014 struktur AG, Dirk Farin <farin@struktur.de>
 *
 * This file is part of libde265.
 *
 * libde265 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * libde265 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libde265.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DE265_SCAN_H
#define DE265_SCAN_H

#include <stdint.h>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef OPT_1
extern const uint8_t* scan_order_y[3][7];
extern const uint8_t* scan_order_x[3][7];
extern const uint8_t* scans_bk[3][6];
extern const uint8_t* scans_pos[3][6];
const uint8_t* get_scan_order_x(int log2BlockSize, int scanIdx);
const uint8_t* get_scan_order_y(int log2BlockSize, int scanIdx);
#endif
typedef struct {
  uint8_t x,y;
} position;

typedef struct {
  uint8_t subBlock;
  uint8_t scanPos;
} scan_position;

#ifndef OPT_1
void init_scan_orders();

/* scanIdx: 0 - diag, 1 - horiz, 2 - verti
 */
const position* get_scan_order(int log2BlockSize, int scanIdx);

scan_position get_scan_position(int x,int y, int scanIdx, int log2BlkSize);
#endif

#endif
