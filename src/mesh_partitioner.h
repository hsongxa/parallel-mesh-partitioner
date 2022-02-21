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
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

#include "hilbert_curve.h"
#include "mpi_datatype_traits.h"

namespace pmp {

// version with no custom weights: k is the desired number of parts to partition into
template<typename MSH, typename OutputItr,
         template<typename> class SFC = hilbert_curve_3d>
int partition(const MSH& mesh, int k, OutputItr it, MPI_Comm comm)
{
  using R = typename MSH::coordinate_type;
  using I = typename MSH::index_type;

  // get the global bounding box
  R min[3], max[3];
  std::tuple<R, R, R, R, R, R> lbbox = mesh.local_bounding_box();
  R lmin[3] = {std::get<0>(lbbox), std::get<1>(lbbox), std::get<2>(lbbox)};
  R lmax[3] = {std::get<3>(lbbox), std::get<4>(lbbox), std::get<5>(lbbox)};
  MPI_Allreduce(lmin, min, 3, mpi_datatype_v<R>, MPI_MIN, comm);
  MPI_Allreduce(lmax, max, 3, mpi_datatype_v<R>, MPI_MAX, comm);

  // count number of cells in each coarse bin
  int num_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes); // p (# of processes) must be 8^n

  I num_local_cells = mesh.num_local_cells();
  std::vector<I>   sending_scheme(num_processes + 1, 0); // allocate one extra space as this will store offsets later
  std::vector<int> sfc_coarse_indices(num_local_cells);

  int coarse_recursion_depth = 0;
  int num_coarse_bins = num_processes;
  while (num_coarse_bins > 7) { num_coarse_bins >>= 3; coarse_recursion_depth++; }
  num_coarse_bins = static_cast<int>(std::pow(8, coarse_recursion_depth));

  SFC<R> sfc(min[0], min[1], min[2], max[0], max[1], max[2]);
  for (I c = 0; c < num_local_cells; ++c)
  {
    std::tuple<R, R, R> centroid = mesh.cell_centroid(c);
    std::int64_t coarse_index = sfc.index(std::get<0>(centroid), std::get<1>(centroid), std::get<2>(centroid),
                                          coarse_recursion_depth);
    assert(coarse_index >= 0 && coarse_index < num_coarse_bins);
    sending_scheme[coarse_index + 1]++;
    sfc_coarse_indices[c] = static_cast<int>(coarse_index);
  }

  // communicate the sending-receiving schemes
  std::vector<I> receiving_scheme(num_processes + 1, 0); // allocate one extra space as this will store offsets later
  MPI_Alltoall(sending_scheme.data() + 1, 1, mpi_datatype_v<I>, receiving_scheme.data() + 1, 1, mpi_datatype_v<I>, comm);

  // compute offsets to send/recv buffers
  int num_send_processes = 0;
  assert(sending_scheme[0] == 0);
  for (std::size_t i = 1; i < sending_scheme.size(); ++i)
  {
    if (sending_scheme[i] > 0) num_send_processes++;
    sending_scheme[i] += sending_scheme[i - 1]; // change the values to be offsets to the send_buffer
  }

  int num_recv_processes = 0;
  assert(receiving_scheme[0] == 0);
  for (std::size_t i = 1; i < receiving_scheme.size(); ++i)
  {
    if (receiving_scheme[i] > 0) num_recv_processes++;
    receiving_scheme[i] += receiving_scheme[i - 1]; // change the values to be offsets to the recv_buffer
  }

  // point-to-point communications: receiving
  std::vector<std::int64_t> recv_buffer(receiving_scheme[receiving_scheme.size() - 1]);
  std::vector<MPI_Request>  recv_requests(num_recv_processes);
  num_recv_processes = 0; // re-use this variable as index to recv_requests/statuses
  for (std::size_t p = 0; p < receiving_scheme.size() - 1; ++p)
  {
    I count = receiving_scheme[p + 1] - receiving_scheme[p];
    if (count > 0)
      MPI_Irecv(recv_buffer.data() + receiving_scheme[p], count, mpi_datatype_v<std::int64_t>, 
                p, 0, comm, &recv_requests[num_recv_processes++]);
  }

  // point-to-point communications: sending
  std::vector<std::int64_t> send_buffer(sending_scheme[sending_scheme.size() - 1]);
  std::vector<MPI_Request>  send_requests(num_send_processes);
  num_send_processes = 0; // re-use this variable as index to send_requests/statuses
  for (std::size_t p = 0; p < sending_scheme.size() - 1; ++p)
  {
    I count = sending_scheme[p + 1] - sending_scheme[p];
    if (count > 0)
    {
      I index = sending_scheme[p];
      for (I c = 0; c < num_local_cells; ++c)
      {
        if (sfc_coarse_indices[c] == static_cast<int>(p)) // NOTE: assume process' rank is the same as coarse bin index
        {
          std::tuple<R, R, R> centroid = mesh.cell_centroid(c);
          send_buffer[index++] = sfc.index(std::get<0>(centroid), std::get<1>(centroid), std::get<2>(centroid));
        }
      }
      MPI_Isend(send_buffer.data() + sending_scheme[p], count, mpi_datatype_v<std::int64_t>,
                p, 0, comm, &send_requests[num_send_processes++]);
    }
  }

  int count = send_requests.size();
  std::vector<MPI_Status> send_statuses(count);
  MPI_Waitall(count, send_requests.data(), send_statuses.data());
  count = recv_requests.size();
  std::vector<MPI_Status> recv_statuses(count);
  MPI_Waitall(count, recv_requests.data(), recv_statuses.data());

  // "Allgather" coarse bin weights before partitioning, i.e., assigning parts for each received cell
  std::vector<I> coarse_bin_weights(num_coarse_bins, 0);
  MPI_Allgather(receiving_scheme.data() + receiving_scheme.size() - 1, 1, mpi_datatype_v<I>,
                coarse_bin_weights.data(), 1, mpi_datatype_v<I>, comm);

  // partition - sort received sfc indices
  // TODO: this step can be overlapped with backward point-to-point communication as well - see below
  std::vector<std::size_t> sorted_indices(recv_buffer.size());
  for (std::size_t i = 0 ; i < sorted_indices.size(); ++i)
    sorted_indices[i] = i;
  // TODO: parallelize the sorting
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&](std::size_t a, std::size_t b){ return recv_buffer[a] < recv_buffer[b]; });

  // partition - part assignment is overlapped with the backward point-to-point communication below
  // here we only determine the starting partition of this coarse bin
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  I prev_weights = 0, total_weights = 0;
  for (std::size_t i = 0; i < coarse_bin_weights.size(); ++i)
  {
    I weight = coarse_bin_weights[i];
    if (i < rank) prev_weights += weight;
    total_weights += weight;
  }
  I part_size = std::round(static_cast<double>(total_weights) / static_cast<double>(k));
  if (part_size == 0) part_size = 1; // skew toward lower ranks
  int init_part = prev_weights / part_size;
  I residual = part_size - (prev_weights % part_size);

  // backward point-to-point communication: receiving (re-use the send_buffer)
  num_send_processes = 0;
  for (std::size_t p = 0; p < sending_scheme.size() - 1; ++p)
  {
    I count = sending_scheme[p + 1] - sending_scheme[p];
    if (count > 0)
      MPI_Irecv(send_buffer.data() + sending_scheme[p], count, mpi_datatype_v<std::int64_t>, 
                p, 0, comm, &send_requests[num_send_processes++]);
  }

  // backward point-to-point communication: sending (re-use the recv_buffer)
  num_recv_processes = 0;
  for (std::size_t p = 0; p < receiving_scheme.size() - 1; ++p)
  {
    I count = receiving_scheme[p + 1] - receiving_scheme[p];
    if (count > 0)
    {
      I index = receiving_scheme[p];
      for (I i = 0; i < count; ++i)
      {
          std::size_t order = sorted_indices[index + i];
          recv_buffer[index + i] = order < residual ? init_part : (init_part + (order - residual) / part_size + 1);
      }
      MPI_Isend(recv_buffer.data() + index, count, mpi_datatype_v<std::int64_t>,
                p, 0, comm, &recv_requests[num_recv_processes++]);
    }
  }

  MPI_Waitall(num_recv_processes, recv_requests.data(), recv_statuses.data());
  MPI_Waitall(num_send_processes, send_requests.data(), send_statuses.data());

  // populate the results
  recv_buffer.resize(send_buffer.size());
  for (std::size_t p = 0; p < sending_scheme.size() - 1; ++p)
  {
    I count = sending_scheme[p + 1] - sending_scheme[p];
    if (count > 0)
    {
      I index = sending_scheme[p];
      for (I c = 0; c < num_local_cells; ++c)
        if (sfc_coarse_indices[c] == static_cast<int>(p)) // NOTE: assume process' rank is the same as coarse bin index
          recv_buffer[c] = send_buffer[index++];
    }
  }
  for (I c = 0; c < num_local_cells; ++c)
    it = recv_buffer[c];

  return 0;
}


// version with custom cell weights
template<typename MSH, typename WItr, typename OutputItr,
         template<typename> class SFC = hilbert_curve_3d>
int partition(const MSH& mesh, int k, WItr wit, OutputItr oit, MPI_Comm comm)
{

  return 0;
}


} // namespace pmp

#endif
