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

#ifndef SERIAL_TEST_MESH_H
#define SERIAL_TEST_MESH_H

#include <tuple>
#include <cassert>

// A serial mesh used for testing, though the mesh can be instantiated on multiple ranks. Rank 0 holds
// NX x NY x NZ cells and all other ranks hold no cells. This is used to test the special case where
// rank 0 reads in a serial mesh and p (p >= 1) processes are used to partition the mesh into k parts.
//
// This mesh implements the minimum API required by the parallel mesh partitioner.
template<typename R, typename I> // R - vertex coordinate type, I - index type for vertices and cells
class serial_test_mesh
{
public:
  using coordinate_type = R;
  using index_type = I;

public:
  explicit serial_test_mesh(int rank) : NX(10), NY (10), NZ(10), RANK(rank) {}
  serial_test_mesh(I nx, I ny, I nz, int rank) : NX(nx), NY(ny), NZ(nz), RANK(rank) {}

  I num_local_cells() const { return RANK == 0 ? NX * NY * NZ : 0; }

  std::tuple<R, R, R, R, R, R> local_bounding_box() const
  {
    return RANK == 0 ? std::make_tuple(0, 0, 0, NX, NY, NZ) : 
           std::make_tuple(0, 0, 0, 0, 0, 0);
  }

  std::tuple<R, R, R> cell_centroid(I c) const
  {
    assert(c >= 0 && c < num_local_cells()); // this will fail on RANK != 0

    I l = c % (NX * NY);
    R half = static_cast<R>(1) / static_cast<R>(2);
    return std::make_tuple((l % NX) + half, (l / NX) + half, (c / (NX * NY)) + half);
  }

private:
  const I NX;
  const I NY;
  const I NZ;
  const int RANK;
};

#endif
