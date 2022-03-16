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

#include "hilbert_curve.h"
#include "mpi_datatype_traits.h"
#include "mpi_utils.h"

namespace pmp {

// function objects used to populate data sending to the specified rank
template<typename SFC, typename MSH>
struct sfc_index_generator
{
  sfc_index_generator(const SFC& sfc, const MSH& mesh, const int* cell_ranks)
    : m_sfc(&sfc), m_mesh(&mesh), m_cell_ranks(cell_ranks) {}

  template<typename ForwardItr>
  void operator()(int rank, ForwardItr it) const
  {
    for (typename MSH::index_type c = 0; c < m_mesh->num_local_cells(); ++c)
    {
      if (m_cell_ranks[c] == rank)
      {
        auto centroid = m_mesh->cell_centroid(c);
        *it++ = m_sfc->index(std::get<0>(centroid), std::get<1>(centroid), std::get<2>(centroid));
      }
    }
  }

private:
  const SFC* m_sfc;
  const MSH* m_mesh;
  const int* m_cell_ranks; // cells in the local mesh mapped to processes (based on their coarse bin assignment)
};

template<typename IndexType>
struct sfc_equal_weight_partitioner
{
  sfc_equal_weight_partitioner(const std::int64_t* cell_sfc_indices, IndexType cell_count,
                               const IndexType* rank_offsets, int part_begin, double part_size,
                               double remaining_capacity)
    : m_cell_sfc_indices(cell_sfc_indices), m_rank_offsets(rank_offsets),
      m_part_begin(part_begin), m_part_size(part_size), m_remaining_capacity(remaining_capacity),
      m_indices_ordering(cell_count)
  { 
    // proxy sort on sfc indices of cells assigned to this process
    for (IndexType i = 0 ; i < cell_count; ++i) m_indices_ordering[i] = i;
    std::sort(m_indices_ordering.begin(), m_indices_ordering.end(), [&](IndexType a, IndexType b)
              { return m_cell_sfc_indices[a] < m_cell_sfc_indices[b]; });
  }

  template<typename ForwardItr>
  void operator()(int rank, ForwardItr it) const
  {
    IndexType offset = m_rank_offsets[rank];
    for (IndexType i = 0; i < m_rank_offsets[rank + 1] - offset; ++i)
    {
        double order = static_cast<double>(m_indices_ordering[offset + i]);
        *it++ = order < m_remaining_capacity ? m_part_begin :
                (m_part_begin + static_cast<int>((order - m_remaining_capacity) / m_part_size) + 1);
    }
  }

private:
  const std::int64_t* m_cell_sfc_indices; // sfc indices of cells assigned to this process, grouped 
  const IndexType*    m_rank_offsets;     // by ranks from which they are received hence these offsets
  int                 m_part_begin;
  double              m_part_size;
  double              m_remaining_capacity; // leftover capacity of the part_begin
  std::vector<IndexType> m_indices_ordering; // ordering of the sfc indices
};


// version with no custom weights: k is the desired number of parts to partition into
template<typename MSH, typename OutputItr,
         template<typename> class SFC = sfc::hilbert_curve_3d>
int partition(const MSH& mesh, int k, OutputItr it, MPI_Comm comm)
{
  using R = typename MSH::coordinate_type;
  using I = typename MSH::index_type;

  // determine number of (coarse) bins based on number of processes
  int num_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  int bin_depth = 0;
  int num_bins = num_processes;
  while (num_bins > 7) { num_bins >>= 3; bin_depth++; }
  num_bins = static_cast<int>(std::pow(8, bin_depth));

  // get the global bounding box for SFC
  R min[3], max[3];
  std::tuple<R, R, R, R, R, R> lbbox = mesh.local_bounding_box();
  R lmin[3] = {std::get<0>(lbbox), std::get<1>(lbbox), std::get<2>(lbbox)};
  R lmax[3] = {std::get<3>(lbbox), std::get<4>(lbbox), std::get<5>(lbbox)};
  MPI_Allreduce(lmin, min, 3, mpi_datatype_v<R>, MPI_MIN, comm);
  MPI_Allreduce(lmax, max, 3, mpi_datatype_v<R>, MPI_MAX, comm);
  SFC<R> sfc(min[0], min[1], min[2], max[0], max[1], max[2]);

  // evaluate total weight
  I num_local_cells = mesh.num_local_cells();
  I total_weight = 0;

  // associate cells to bins
  std::vector<int> associated_bins(num_local_cells);
  for (I c = 0; c < num_local_cells; ++c)
  {
    std::tuple<R, R, R> centroid = mesh.cell_centroid(c);
    std::int64_t coarse_index = sfc.index(std::get<0>(centroid), std::get<1>(centroid), std::get<2>(centroid),
                                          bin_depth);
    assert(coarse_index >= 0 && coarse_index < num_bins);
    associated_bins[c] = static_cast<int>(coarse_index);
  }

  // communicate the send/recv schemes
  std::vector<I> send_scheme(num_processes + 1, 0); // allocate one extra space as this will store offsets later
  std::vector<I> recv_scheme(num_processes + 1, 0); // allocate one extra space as this will store offsets later
  // TODO: move the allocation of send/recv_schemes to outside the phase loop but re-fill with zeros in every phase
  for (I c = 0; c < num_local_cells; ++c)
  {
    int bin_index = associated_bins[c];
    if (bin_index < num_processes)  // bins that are processed by this phase
      send_scheme[bin_index + 1]++; // are mapped to ranks
  }
  MPI_Alltoall(send_scheme.data() + 1, 1, mpi_datatype_v<I>, recv_scheme.data() + 1, 1, mpi_datatype_v<I>, comm);

  // convert to the form of offsets (to send/recv buffers, respectively)
  int num_send_processes = 0;
  assert(send_scheme[0] == 0);
  for (std::size_t i = 1; i < send_scheme.size(); ++i)
  {
    if (send_scheme[i] > 0) num_send_processes++;
    send_scheme[i] += send_scheme[i - 1];
  }

  int num_recv_processes = 0;
  assert(recv_scheme[0] == 0);
  for (std::size_t i = 1; i < recv_scheme.size(); ++i)
  {
    if (recv_scheme[i] > 0) num_recv_processes++;
    recv_scheme[i] += recv_scheme[i - 1];
  }

  // point-to-point communication of the sfc indices
  std::vector<std::int64_t> send_buffer(send_scheme[send_scheme.size() - 1]);
  std::vector<std::int64_t> recv_buffer(recv_scheme[recv_scheme.size() - 1]);
  utils::point_to_point_communication( send_scheme.begin(), recv_scheme.begin(), send_scheme.size(),
                                       send_buffer.data(), recv_buffer.data(),
                                       sfc_index_generator(sfc, mesh, associated_bins.data()), comm );

  // "Allgather" bin weights before partitioning
  std::vector<I> bin_weights(num_bins, 0);
  MPI_Allgather(recv_scheme.data() + recv_scheme.size() - 1, 1, mpi_datatype_v<I>,
                bin_weights.data(), 1, mpi_datatype_v<I>, comm);

  // partition - part assignment is overlapped with the backward point-to-point communication below
  // here we only determine the starting partition, partition size, ..., etc.
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  I prev_bin_weight = 0;
  for (int i = 0; i < num_bins; ++i)
  {
    I weight = bin_weights[i];
    if (i < rank) prev_bin_weight += weight;
    total_weight += weight;
  }
  double part_size = static_cast<double>(total_weight) / static_cast<double>(k);
  int init_part = static_cast<double>(prev_bin_weight) / part_size;
  double residual = (init_part + 1 ) * part_size - static_cast<double>(prev_bin_weight);

  // backward point-to-point communication of the assigned parts
  utils::point_to_point_communication( recv_scheme.begin(), send_scheme.begin(), recv_scheme.size(),
                                       recv_buffer.data(), send_buffer.data(),
                                       sfc_equal_weight_partitioner(recv_buffer.data(), static_cast<I>(recv_buffer.size()),
                                                                    recv_scheme.data(), init_part, part_size, residual),
                                       comm );

  // populate the results
  recv_buffer.resize(send_buffer.size());
  for (std::size_t p = 0; p < send_scheme.size() - 1; ++p)
  {
    I count = send_scheme[p + 1] - send_scheme[p];
    if (count > 0)
    {
      I index = send_scheme[p];
      for (I c = 0; c < num_local_cells; ++c)
        if (associated_bins[c] == static_cast<int>(p))
          recv_buffer[c] = send_buffer[index++];
    }
  }
  for (I c = 0; c < num_local_cells; ++c) it = recv_buffer[c];

  return 0;
}


// version with custom cell weights
template<typename MSH, typename WItr, typename OutputItr,
         template<typename> class SFC = sfc::hilbert_curve_3d>
int partition(const MSH& mesh, int k, WItr wit, OutputItr oit, MPI_Comm comm)
{

  return 0;
}


} // namespace pmp

#endif
