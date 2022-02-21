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
  using coordinate_type = R;
  using index_type = I;

public:
  explicit distributed_test_mesh(int rank) : NX(10), NY (10), NZ(10), RANK(rank) {}
  distributed_test_mesh(I nx, I ny, I nz, int rank) : NX(nx), NY(ny), NZ(nz), RANK(rank) {}

  I num_local_cells() const { return NX * NY * NZ; }

  std::tuple<R, R, R, R, R, R> local_bounding_box() const
  { return std::make_tuple(RANK * NX, 0, 0, (RANK + 1) * NX, NY, NZ); }

  std::tuple<R, R, R> cell_centroid(I c) const
  {
    assert(c >= 0 && c < num_local_cells());

    I l = c % (NX * NY);
    R half = static_cast<R>(1) / static_cast<R>(2);
    return std::make_tuple(RANK * NX + (l % NX) + half, (l / NX) + half, (c / (NX * NY)) + half);
  }

private:
  const I NX;
  const I NY;
  const I NZ;
  const int RANK;
};

#endif
