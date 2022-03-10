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

#ifndef MPI_UTILS_H
#define MPI_UTILS_H 

#include <mpi.h>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "mpi_datatype_traits.h"

namespace pmp {

// sending/receiving schemes store, for each destination/source process in the communicator,
// the offset to the sending/receiving buffer; also, the generation of the sending data is
// overlapped with the communication
template<typename ConstRandAccItr, typename DataType, typename DataGen>
void point_to_point_communication(ConstRandAccItr sending_scheme_begin,
                                  ConstRandAccItr receiving_scheme_begin,
                                  std::size_t scheme_size,
                                  DataType* sending_buffer_begin,
                                  DataType* receiving_buffer_begin,
                                  const DataGen& data_generator, MPI_Comm comm)
{
  // receiving
  std::size_t num_recvs = 0;
  std::vector<MPI_Request>  recv_requests(scheme_size - 1);
  for (std::size_t p = 0; p < scheme_size - 1; ++p)
  {
    std::size_t count = *(receiving_scheme_begin + p + 1) - *(receiving_scheme_begin + p);
    if (count > 0)
      MPI_Irecv(receiving_buffer_begin + *(receiving_scheme_begin + p), count, mpi_datatype_v<DataType>, 
                p, 0, comm, &recv_requests[num_recvs++]);
  }

  // sending
  std::size_t num_sends = 0;
  std::vector<MPI_Request>  send_requests(scheme_size - 1);
  for (std::size_t p = 0; p < scheme_size - 1; ++p)
  {
    std::size_t count = *(sending_scheme_begin + p + 1) - *(sending_scheme_begin + p);
    if (count > 0)
    {
      DataType* begin = sending_buffer_begin + *(sending_scheme_begin + p);
      data_generator(p, begin); // populate this section of the send buffer
      MPI_Isend(begin, count, mpi_datatype_v<DataType>, p, 0, comm, &send_requests[num_sends++]);
    }
  }

  std::vector<MPI_Status> send_statuses(num_sends);
  MPI_Waitall(num_sends, send_requests.data(), send_statuses.data());
  std::vector<MPI_Status> recv_statuses(num_recvs);
  MPI_Waitall(num_recvs, recv_requests.data(), recv_statuses.data());
}

} // namespace pmp

#endif
