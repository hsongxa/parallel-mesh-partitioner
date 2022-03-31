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
#include <fstream>
#include <cstdio> // use <format> instead for c++20
#include <chrono>

#include <mpi.h>

#include "test_hilbert_curve.h"
#include "serial_test_mesh.h"
#include "distributed_test_mesh.h"
#include "mesh_partitioner.h"

int main (int argc, char* argv[])
{
  // uncomment to run the hilbert curve unit tests
  //test_hilbert_curve();

  int k = 64; // the default number of parts to partition into
  if (argc > 1) k = std::atoi(argv[1]);

  MPI_Init(NULL, NULL);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  serial_test_mesh<float, int> smesh(rank);
  std::vector<int> serial_output;
  pmp::partition(smesh, k, std::back_inserter(serial_output), MPI_COMM_WORLD);
  if (rank == 0)
  {
    char serial_buff[100];
    std::snprintf(serial_buff, sizeof(serial_buff), "PartitionSerial_%d-%d.txt", size, k); // use std::format() instead for c++20
    std::ofstream serial_file(serial_buff);
    serial_file << "x     y     z     p" << std::endl;
    for (int c = 0; c < smesh.num_local_cells(); ++c)
    {
      std::tuple<float, float, float> centroid = smesh.cell_centroid(c); 
      serial_file << std::get<0>(centroid) << " " << std::get<1>(centroid) << " ";
      serial_file << std::get<2>(centroid) << " " << serial_output[c] << std::endl;
    }
  }

  distributed_test_mesh<double, int> dmesh(10, 10, 10, rank);
  auto t0 = std::chrono::system_clock::now();
  std::vector<int> output;
  pmp::partition(dmesh, k, std::back_inserter(output), MPI_COMM_WORLD);
  auto t1 = std::chrono::system_clock::now();
  std::cout << "time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << std::endl;

  // output the partition info
  char buff[100];
  std::snprintf(buff, sizeof(buff), "PartitionDistributed_%d-%d_%d.txt", size, k, rank); // use std::format() instead for c++20
  std::ofstream file(buff);
  file << "x     y     z     p" << std::endl;
  for (int c = 0; c < dmesh.num_local_cells(); ++c)
  {
    std::tuple<double, double, double> centroid = dmesh.cell_centroid(c); 
    file << std::get<0>(centroid) << " " << std::get<1>(centroid) << " " << std::get<2>(centroid) << " " << output[c] << std::endl;
  }

  MPI_Finalize();

  return 0;
}

