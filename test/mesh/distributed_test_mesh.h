/*
  Copyright (C) 2022 Hao Song

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef DISTRIBUTED_TEST_MESH_H
#define DISTRIBUTED_TEST_MESH_H

#include <tuple>
#include <cassert>

// A simple distributed mesh used for testing. Each rank holds NX x NY x NZ cells as its local part
// and each cell is a unit cube. Between ranks, the parts are shifted in the +X direction by NX, i.e.,
// the minimum X coordinate of the part is 0 on rank 0, NX on rank 1, 2 x NX on rank 2, ..., etc.
//
// This mesh implements the minimum API required by the parallel mesh partitioner.
template<typename R, typename I> // R - vertex coordinate type, I - index type for vertices and cells
class distributed_test_mesh
{
public:
  distributed_test_mesh(int rank) : _offset_X(rank * NX) {}

  // number of LOCAL cells!
  I num_cells() const { return NX * NY * NZ; }

  std::tuple<R, R, R, R, R, R> get_bounding_box() const
  { return std::make_tuple(_offset_X, R(0), R(0), _offset_X + NX, NY, NZ); }

  std::tuple<R, R, R> get_cell_centroid(I c) const
  {
    assert(c >= 0 && c < num_cells());

    I l = c % (NX * NY);
    R half = static_cast<R>(1) / static_cast<R>(2);
    return std::make_tuple(_offset_X + (l % NX) + half, (l / NX) + half, (c / (NX * NY)) + half);
  }

private:
  static constexpr I NX = 10;
  static constexpr I NY = 10;
  static constexpr I NZ = 10;
  const R _offset_X;
};

#endif
