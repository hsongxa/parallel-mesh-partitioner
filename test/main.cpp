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

#include <iostream>
#include <mpi.h>

#include "hilbert_curve.h"

int main (int argc, char* argv[])
{
  sfc::hilbert_curve_3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);

  sfc::hilbert_curve_3d((float)0.0, (float)0.0, (float)0.0, (float)1.0, (float)1.0, (float)1.0);

  MPI_Init(NULL, NULL);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "Hello, World from process " << rank << "!" << std::endl;

  MPI_Finalize();

  return 0;
}

