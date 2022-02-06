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

#include <vector>
#include <iterator>
#include <tuple>
#include <iostream>
#include <mpi.h>

#include "unit_tests.h"
#include "distributed_test_mesh.h"
#include "mesh_partitioner.h"

int main (int argc, char* argv[])
{
//  test_hilbert_curve();

  MPI_Init(NULL, NULL);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  distributed_test_mesh<double, int> mesh(rank);
  std::tuple<double, double, double> centroid = mesh.get_cell_centroid(0); 

//  std::cout << "centroid of first cell on process " << rank << ": (";
//  std::cout << std::get<0>(centroid) << ", " << std::get<1>(centroid) << ", " << std::get<2>(centroid) << ")" << std::endl;

  std::vector<int> output;
  pmp::partition(mesh, size, std::back_inserter(output), MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}

