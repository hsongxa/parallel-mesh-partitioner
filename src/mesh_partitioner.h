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

#ifndef MESH_PARTITIONER_H
#define MESH_PARTITIONER_H 

#include <mpi.h>

#include "hilbert_curve.h"

namespace pmp {

// version with no custom weights: k is the desired number of parts to partition into
template<typename MSH, typename OutputItr,
         template<typename> class SFC = hilbert_curve_3d>
int partition(const MSH& mesh, int k, OutputItr it, MPI_Comm comm)
{

  return 0;
}


// version with cell weights
template<typename MSH, typename WItr, typename OutputItr,
         template<typename> class SFC = hilbert_curve_3d>
int partition(const MSH& mesh, int k, WItr wit, OutputItr oit, MPI_Comm comm)
{

  return 0;
}


} // namespace pmp

#endif
