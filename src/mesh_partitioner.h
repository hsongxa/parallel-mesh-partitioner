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
#include <type_traits>
#include <iterator>
#include <numeric>
#include <cassert>

#include "hilbert_curve.h"
#include "mpi_datatype_traits.h"
#include "mpi_utils.h"

namespace pmp {

// policy classes for handling constant vs custom cell weights
template<typename WT> // weight_type
struct constant_weights
{
  using weight_type = WT;

  static constexpr bool is_constant_weight = true;

  explicit constant_weights(WT local_weight) : m_local_weight(local_weight) {}

  weight_type local_weight() const { return m_local_weight; }

private:
  const WT m_local_weight;
};

template<typename ConstRandAccItr>
struct custom_weights
{
  using weight_type = typename std::iterator_traits<ConstRandAccItr>::value_type;

  static constexpr bool is_constant_weight = false;

  custom_weights(ConstRandAccItr wit, std::size_t size) : m_wit(wit), m_size(size) {}

  weight_type local_weight() const { return std::accumulate(m_wit, m_wit + m_size, 0); }

  ConstRandAccItr weights_begin() const { return m_wit; }

  weight_type weight(std::size_t i) const { return *(m_wit + i); }

private:
  // note that these weights could belong to either the local cells or
  // those cells associated a particular bin (after MPI communications) 
  const ConstRandAccItr m_wit;
  const std::size_t m_size;
};

// function objects used to populate data sending to the specified rank
template<typename SFC, typename MSH>
struct sfc_index_generator
{
  sfc_index_generator(const SFC& sfc, const MSH& mesh, const int* cell_bins, int rank_to_bin_offset)
    : m_sfc(&sfc), m_mesh(&mesh), m_cell_bins(cell_bins), m_rank_to_bin_offset(rank_to_bin_offset) {}

  template<typename ForwardItr>
  void operator()(int rank, ForwardItr it) const
  {
    for (typename MSH::index_type c = 0; c < m_mesh->num_local_cells(); ++c)
      if (m_cell_bins[c] == (rank + m_rank_to_bin_offset))
      {
        auto centroid = m_mesh->cell_centroid(c);
        *it++ = m_sfc->index(std::get<0>(centroid), std::get<1>(centroid), std::get<2>(centroid));
      }
  }

private:
  const SFC* m_sfc;
  const MSH* m_mesh;
  const int* m_cell_bins; // cells' bin assignments
  const int  m_rank_to_bin_offset; // mapps ranks to bins for a particular phase
};

template<typename ConstRandAccItr, typename MSH>
struct weight_populator
{
  weight_populator(ConstRandAccItr wit, const MSH& mesh, const int* cell_bins, int rank_to_bin_offset)
    : m_wit(wit), m_mesh(&mesh), m_cell_bins(cell_bins), m_rank_to_bin_offset(rank_to_bin_offset) {}

  template<typename ForwardItr>
  void operator()(int rank, ForwardItr it) const
  {
    for (typename MSH::index_type c = 0; c < m_mesh->num_local_cells(); ++c)
      if (m_cell_bins[c] == (rank + m_rank_to_bin_offset))
        *it++ = *(m_wit + c); // weights are ordered the same way as the local cells of the mesh
  }

private:
  const ConstRandAccItr m_wit;
  const MSH* m_mesh;
  const int* m_cell_bins; // cells' bin assignments
  const int  m_rank_to_bin_offset; // mapps ranks to bins for a particular phase
};

template<typename IndexType, typename WP> // WP - weight policy
struct sfc_partitioner
{
  sfc_partitioner(const std::int64_t* cell_sfc_indices, IndexType cell_count,
                  const IndexType* rank_offsets, int part_begin, double part_size,
                  double remaining_capacity, const WP& weight_policy)
    : m_cell_count(cell_count), m_cell_sfc_indices(cell_sfc_indices), m_rank_offsets(rank_offsets),
      m_part_begin(part_begin), m_part_size(part_size), m_remaining_capacity(remaining_capacity),
      m_weight_policy(&weight_policy), m_sorted_indices(cell_count)
  { 
    // proxy sort on sfc indices of cells assigned to this bin
    for (IndexType i = 0 ; i < cell_count; ++i) m_sorted_indices[i] = i;
    std::sort(m_sorted_indices.begin(), m_sorted_indices.end(), [=](IndexType a, IndexType b)
              { return m_cell_sfc_indices[a] < m_cell_sfc_indices[b]; });
  }

  template<typename RandAccItr>
  void operator()(int rank, RandAccItr it) const
  {
    IndexType rank_begin = m_rank_offsets[rank];
    IndexType rank_end = m_rank_offsets[rank + 1];
    for (IndexType i = 0; i < m_cell_count; ++i)
    {
      IndexType id = m_sorted_indices[i];
      if (id >= rank_begin && id < rank_end)
      {
        if constexpr(m_weight_policy->is_constant_weight)
          *(it + id - rank_begin) = i < m_remaining_capacity ? m_part_begin :
                                    (m_part_begin + static_cast<int>((i - m_remaining_capacity) / m_part_size) + 1);
        else
        {
          // assume weights from the weight policy are in the same
          // order as cells assigned to this bin
          double weight = static_cast<double>(m_weight_policy->weight(id));
          assert(weight > 0);
          if (weight <= m_remaining_capacity)
            m_remaining_capacity -= weight;
          else
          {
            // partition numbers need to be continuous, so even if
            // the weight is so huge that it exceeds m_part_size
            // it is still assigned to the immediate next part, i.e.,
            // there is no skipping of part numbers
            m_remaining_capacity += (m_part_size - weight);
            ++m_part_begin;
          }

          // TODO: round-off errors might cause m_part_begin to pass
          // TODO: the allowed maximum - cap to be below m_part_end?
          *(it + id - rank_begin) = m_part_begin;
        }
      }
    }
  }

private:
  IndexType           m_cell_count;       // number of cells assigned to this bin
  const std::int64_t* m_cell_sfc_indices; // sfc indices of cells assigned to this bin, grouped by
  const IndexType*    m_rank_offsets;     // ranks from which they are received hence these offsets
  mutable int         m_part_begin;
  mutable double      m_part_size;
  mutable double      m_remaining_capacity; // leftover capacity of the part "part_begin" for this bin
  const WP*           m_weight_policy;
  std::vector<IndexType> m_sorted_indices;  // sorted sfc indices in this separate container
};


// MSH - mesh, WP - weight policy, SFC - space filling curve
template<typename MSH, typename WP, typename OutputItr,
         template<typename> class SFC = sfc::hilbert_curve_3d>
void partition_impl(const MSH& mesh, int k, const WP& weight_policy, OutputItr oit, MPI_Comm comm)
{
  using R = typename MSH::coordinate_type;
  using I = typename MSH::index_type;
  using W = typename WP::weight_type;
  static_assert(std::is_arithmetic_v<R>, "wrong coordinate_type");
  static_assert(std::is_integral_v<I>, "wrong index_type");
  static_assert(std::is_arithmetic_v<W>, "wrong weight_type");
  static_assert((std::is_signed_v<I> && sizeof(I) <= sizeof(std::int64_t)) ||
                (std::is_unsigned_v<I> && sizeof(I) < sizeof(std::int64_t)), "index_type exceeds the range of int64_t");

  // determine number of (coarse) bins based on number of processes
  int num_processes, rank;
  MPI_Comm_size(comm, &num_processes);
  MPI_Comm_rank(comm, &rank);

  int bin_depth = 1, num_bins = 8;
  while (num_bins < num_processes) { num_bins <<= 3; bin_depth++; }
  assert(bin_depth <= 10); // num_bins must not overflow the range of int

  // get the global bounding box for SFC
  R min[3], max[3];
  std::tuple<R, R, R, R, R, R> lbbox = mesh.local_bounding_box();
  R lmin[3] = {std::get<0>(lbbox), std::get<1>(lbbox), std::get<2>(lbbox)};
  R lmax[3] = {std::get<3>(lbbox), std::get<4>(lbbox), std::get<5>(lbbox)};
  MPI_Allreduce(lmin, min, 3, mpi_datatype_v<R>, MPI_MIN, comm);
  MPI_Allreduce(lmax, max, 3, mpi_datatype_v<R>, MPI_MAX, comm);
  SFC<R> sfc(min[0], min[1], min[2], max[0], max[1], max[2]);

  // associate cells to bins
  I num_local_cells = mesh.num_local_cells();
  std::vector<int> associated_bins(num_local_cells);
  for (I c = 0; c < num_local_cells; ++c)
  {
    std::tuple<R, R, R> centroid = mesh.cell_centroid(c);
    std::int64_t coarse_index = sfc.index(std::get<0>(centroid), std::get<1>(centroid), std::get<2>(centroid),
                                          bin_depth);
    assert(coarse_index >= 0 && coarse_index < num_bins);
    associated_bins[c] = static_cast<int>(coarse_index);
  }

  // evaluate the total weight
  W total_weight, local_weight = weight_policy.local_weight();
  MPI_Allreduce(&local_weight, &total_weight, 1, mpi_datatype_v<W>, MPI_SUM, comm);

  // phase loop
  std::vector<int> results(num_local_cells);
  std::vector<I> send_scheme(num_processes + 1); // need one extra space as it will be transferred to offsets later
  std::vector<I> recv_scheme(num_processes + 1); // need one extra space as it will be transferred to offsets later
  std::vector<W> bin_weights(num_processes);
  W phase_weight, prev_phase_weight = 0;
  for(int bin_begin = 0; bin_begin < num_bins; bin_begin += num_processes)
  {
    int bin_end = bin_begin + num_processes;
    if (bin_end > num_bins) bin_end = num_bins;

    std::fill(send_scheme.begin(), send_scheme.end(), 0);
    std::fill(recv_scheme.begin(), recv_scheme.end(), 0);
    
    // communicate the send/recv schemes
    for (I c = 0; c < num_local_cells; ++c)
    {
      int bin_index = associated_bins[c];
      if (bin_index >= bin_begin && bin_index < bin_end)  // bins that are processed by this phase
        send_scheme[bin_index - bin_begin + 1]++;         // are mapped to ranks
    }
    MPI_Alltoall(send_scheme.data() + 1, 1, mpi_datatype_v<I>, recv_scheme.data() + 1, 1, mpi_datatype_v<I>, comm);

    // convert to the form of offsets to send/recv buffers
    int num_send_processes = 0, num_recv_processes = 0;
    assert(send_scheme[0] == 0 && recv_scheme[0] == 0);
    for (std::size_t i = 1; i < send_scheme.size(); ++i)
    {
      if (send_scheme[i] > 0) num_send_processes++;
      if (recv_scheme[i] > 0) num_recv_processes++;
      send_scheme[i] += send_scheme[i - 1];
      recv_scheme[i] += recv_scheme[i - 1];
    }

    // point-to-point communication of the sfc indices
    std::vector<std::int64_t> send_buffer(send_scheme[num_processes]);
    std::vector<std::int64_t> recv_buffer(recv_scheme[num_processes]);
    utils::point_to_point_communication( send_scheme.begin(), recv_scheme.begin(), send_scheme.size(),
                                         send_buffer.data(), recv_buffer.data(),
                                         sfc_index_generator(sfc, mesh, associated_bins.data(), bin_begin), comm );

    if constexpr (weight_policy.is_constant_weight)
    {
      // "Allgather" bin weights before partitioning
      MPI_Allgather(recv_scheme.data() + num_processes, 1, mpi_datatype_v<I>, bin_weights.data(), 1, mpi_datatype_v<W>, comm);

      // partition - part assignment is overlapped with the backward point-to-point communication below
      // here we only determine the starting partition, partition size, ..., etc.
      W prev_bin_weight = prev_phase_weight;
      phase_weight = 0;
      for (int p = 0; p < num_processes; ++p)
      {
        W weight = bin_weights[p];
        if (p < rank) prev_bin_weight += weight;
        phase_weight += weight;
      }
      // note that we can now convert the weight_type to types of our choice when expressing partitions
      // (int/double for init_part/residual, respectively) -- this is necessary when weight_type is an
      // integral type which will result in strongly skewed partitions if we do not do such conversions;
      // on the other hand, hopefully, weight_type is not a long double so these conversions do not
      // loose precisions either!
      static_assert(std::is_integral_v<W> || (std::is_floating_point_v<W> && sizeof(W) <= sizeof(double)));
      double part_size = static_cast<double>(total_weight) / k;
      int init_part = static_cast<double>(prev_bin_weight) / part_size;
      double residual = (init_part + 1 ) * part_size - static_cast<double>(prev_bin_weight);

      // backward point-to-point communication of the assigned parts
      utils::point_to_point_communication( recv_scheme.begin(), send_scheme.begin(), recv_scheme.size(),
                                           recv_buffer.data(), send_buffer.data(),
                                           sfc_partitioner(recv_buffer.data(), static_cast<I>(recv_buffer.size()),
                                                           recv_scheme.data(), init_part, part_size, residual,
                                                           // note that this is a new constant_weights policy that
                                                           // does not really matter, however
                                                           constant_weights(recv_buffer.size())),
                                           comm );
    }
    else
    {
      // need another point-to-point communication of weights first
      std::vector<W> weights_send_buffer(send_scheme[num_processes]);
      std::vector<W> weights_recv_buffer(recv_scheme[num_processes]);
      utils::point_to_point_communication( send_scheme.begin(), recv_scheme.begin(), send_scheme.size(),
                                           weights_send_buffer.data(), weights_recv_buffer.data(),
                                           weight_populator(weight_policy.weights_begin(), mesh, associated_bins.data(), bin_begin),
                                           comm );

      // "Allgather" bin weights before partitioning
      W bin_weight = std::accumulate(weights_recv_buffer.begin(), weights_recv_buffer.end(), 0);
      MPI_Allgather(&bin_weight, 1, mpi_datatype_v<W>, bin_weights.data(), 1, mpi_datatype_v<W>, comm);

      // partition - part assignment is overlapped with the backward point-to-point communication below
      // here we only determine the starting partition, partition size, ..., etc.
      W prev_bin_weight = prev_phase_weight;
      phase_weight = 0;
      for (int p = 0; p < num_processes; ++p)
      {
        W weight = bin_weights[p];
        if (p < rank) prev_bin_weight += weight;
        phase_weight += weight;
      }
      // note that we can now convert the weight_type to types of our choice when expressing partitions
      // (int/double for init_part/residual, respectively) -- this is necessary when weight_type is an
      // integral type which will result in strongly skewed partitions if we do not do such conversions;
      // on the other hand, hopefully, weight_type is not a long double so these conversions do not
      // loose precisions either!
      static_assert(std::is_integral_v<W> || (std::is_floating_point_v<W> && sizeof(W) <= sizeof(double)));
      double part_size = static_cast<double>(total_weight) / k;
      int init_part = static_cast<double>(prev_bin_weight) / part_size;
      double residual = (init_part + 1 ) * part_size - static_cast<double>(prev_bin_weight);

      // backward point-to-point communication of the assigned parts
      utils::point_to_point_communication( recv_scheme.begin(), send_scheme.begin(), recv_scheme.size(),
                                           recv_buffer.data(), send_buffer.data(),
                                           sfc_partitioner(recv_buffer.data(), static_cast<I>(recv_buffer.size()),
                                                           recv_scheme.data(), init_part, part_size, residual,
                                                           // note that this is a new custom_weights policy working with
                                                           // weights assoicated to the bin that this rank is responsible for
                                                           custom_weights(weights_recv_buffer.begin(), weights_recv_buffer.size())),
                                           comm );
    }

    // populate the results obtained from this phase
    for (int p = 0; p < num_processes; ++p)
    {
      I count = send_scheme[p + 1] - send_scheme[p];
      if (count > 0)
      {
        I index = send_scheme[p];
        for (I c = 0; c < num_local_cells; ++c)
          if (associated_bins[c] == (static_cast<int>(p) + bin_begin)) // map back ranks to bins
            results[c] = send_buffer[index++];
      }
    }

    prev_phase_weight += phase_weight;
  }

  for (I c = 0; c < num_local_cells; ++c) oit = results[c];
}

// version with no custom weights: k is the desired number of parts to partition into
template<typename MSH, typename OutputItr>
void partition(const MSH& mesh, int k, OutputItr it, MPI_Comm comm)
{ partition_impl(mesh, k, constant_weights(mesh.num_local_cells()), it, comm); }

// version with custom cell weights
template<typename MSH, typename ConstRandAccItr, typename OutputItr>
void partition(const MSH& mesh, int k, ConstRandAccItr wit, OutputItr oit, MPI_Comm comm)
{ partition_impl(mesh, k, custom_weights(wit, mesh.num_local_cells()), oit, comm); }

} // namespace pmp

#endif
