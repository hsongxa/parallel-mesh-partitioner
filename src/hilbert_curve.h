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

#ifndef HILBERT_CURVE_H
#define HILBERT_CURVE_H

#include <cassert>
#include <cinttypes>
#include <tuple>

namespace sfc {

// adapted from "Programming the Hilbert Curve", John Skilling, 2004,
// Baysian Inference and Maximum Entropy Methods in Science and Engineering:
// 23rd International Workshop, edited by G. Erickson and Y. Zhai

template<typename UINT, int DIM>
void transpose_to_axes(UINT* x, int b = sizeof(UINT) * 8)
{
  assert(b > 0 && b <= static_cast<int>(sizeof(UINT) * 8));
  UINT N = (UINT)2 << (b - 1);

  // Gray decode by H ^ (H/2)
  UINT t = x[DIM - 1] >> 1;
  for (int i = DIM - 1; i > 0; i--) x[i] ^= x[i - 1];
  x[0] ^= t;

  // undo excess work
  for (UINT Q = 2; Q != N; Q <<= 1)
  {
    UINT P = Q - 1;
    for (int i = DIM - 1; i >= 0; i--)
      if (x[i] & Q) x[0] ^= P; // invert
      else // exchange
      {
        t = (x[0] ^ x[i]) & P;
        x[0] ^= t;
        x[i] ^= t;
      }
  }
}

template<typename UINT, int DIM>
void axes_to_transpose(UINT* x, int b = sizeof(UINT) * 8)
{
  assert(b > 0 && b <= static_cast<int>(sizeof(UINT) * 8));
  UINT M = (UINT)1 << (b - 1);
  UINT t;

  // inverse undo
  for (UINT Q = M; Q > 1; Q >>= 1)
  {
    UINT P = Q - 1;
    for (int i = 0; i < DIM; i++)
      if (x[i] & Q) x[0] ^= P; // invert
      else // exchange
      {
        t = (x[0] ^ x[i]) & P;
        x[0] ^= t;
        x[i] ^= t;
      }
  }

  // Gray encode
  for (int i = 1; i < DIM; i++) x[i] ^= x[i - 1];
  t = 0;
  for (UINT Q = M; Q > 1; Q >>= 1)
    if (x[DIM - 1] & Q) t ^= Q - 1;
  for (int i = 0; i < DIM; i++) x[i] ^= t;
}

// 3D hilbert curve encoded in one int_64: maximum 21-level recursions in each dimension
template <typename R>
class hilbert_curve_3d
{
public:
	hilbert_curve_3d(R x_min, R y_min, R z_min, R x_max, R y_max, R z_max)
		: _homothety{ x_min < x_max ? (R)(1.0L) / (x_max - x_min) : (R)(1.0L),
                              y_min < y_max ? (R)(1.0L) / (y_max - y_min) : (R)(1.0L),
                              z_min < z_max ? (R)(1.0L) / (z_max - z_min) : (R)(1.0L) },
		  _translation{ -x_min, -y_min, -z_min } {}

	// hilbert index for an arbitrary point located in the defined extent
	std::int64_t index(R x, R y, R z, int recursion_depth = _max_recursion_depth) const;

	// the (approximate) location of the point corresponding to the given hilbert index
	std::tuple<R, R, R> coords(std::int64_t index, int recursion_depth = _max_recursion_depth) const;

private:
	// maximum index in one dimension: 2^21 - 1 = 2097151
        static constexpr int  _max_recursion_depth = 21;
        const R               _homothety[3];
	const R               _translation[3];
	mutable int	      _cache[3];
};

template <typename R>
std::int64_t hilbert_curve_3d<R>::index(R x, R y, R z, int recursion_depth) const
{
  assert(recursion_depth > 0 && recursion_depth <= _max_recursion_depth);

  int max_1d_index = (1 << recursion_depth) - 1;
  _cache[0] = static_cast<int>((x + _translation[0]) * _homothety[0] * max_1d_index);
  _cache[1] = static_cast<int>((y + _translation[1]) * _homothety[1] * max_1d_index);
  _cache[2] = static_cast<int>((z + _translation[2]) * _homothety[2] * max_1d_index);
  assert(_cache[0] >= 0 && _cache[0] <= max_1d_index);
  assert(_cache[1] >= 0 && _cache[1] <= max_1d_index);
  assert(_cache[2] >= 0 && _cache[2] <= max_1d_index);

  axes_to_transpose<int, 3>(_cache, recursion_depth);

  // interleave the bits of transpose into one integer
  std::int64_t h = 0;
  for (std::int64_t b = 1, k = 2; k <= 2 * recursion_depth; b <<= 1, k += 2) h ^= ((static_cast<std::int64_t>(_cache[0]) & b) << k);
  for (std::int64_t b = 1, k = 1; k < 2 * recursion_depth; b <<= 1, k += 2) h ^= ((static_cast<std::int64_t>(_cache[1]) & b) << k);
  for (std::int64_t b = 1, k = 0; k < 2 * recursion_depth; b <<= 1, k += 2) h ^= ((static_cast<std::int64_t>(_cache[2]) & b) << k);
  return h;
}

template <typename R>
std::tuple<R, R, R> hilbert_curve_3d<R>::coords(std::int64_t index, int recursion_depth) const
{
  assert(index >= 0);
  assert(recursion_depth > 0 && recursion_depth <= _max_recursion_depth);

  // un-interleave the bits
  _cache[0] = 0;
  for (std::int64_t b = 4, k = 2; k <= 2 * recursion_depth; b <<= 3, k += 2) _cache[0] ^= ((index & b) >> k);
  _cache[1] = 0;
  for (std::int64_t b = 2, k = 1; k < 2 * recursion_depth; b <<= 3, k += 2) _cache[1] ^= ((index & b) >> k);
  _cache[2] = 0;
  for (std::int64_t b = 1, k = 0; k < 2 * recursion_depth; b <<= 3, k += 2) _cache[2] ^= ((index & b) >> k);

  transpose_to_axes<int, 3>(_cache, recursion_depth);

  int max_1d_index = (1 << recursion_depth) - 1;
  return std::make_tuple( static_cast<R>(_cache[0]) / static_cast<R>(max_1d_index) / _homothety[0] - _translation[0],
                          static_cast<R>(_cache[1]) / static_cast<R>(max_1d_index) / _homothety[1] - _translation[1],
                          static_cast<R>(_cache[2]) / static_cast<R>(max_1d_index) / _homothety[2] - _translation[2] );
}

} // namespace pmp

#endif
