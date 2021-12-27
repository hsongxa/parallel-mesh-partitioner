#include "hilbert_curve.h"

// TODO: there might exist a more efficient implementation which directly works with
// the interleaved integer by customizing the original axes_to_transpose() code
std::int64_t hilbert_curve_3d::index(double x, double y, double z) const
{
	_cache[0] = static_cast<int>((x + _translation[0]) * _homothety[0] * _max_size);
	_cache[1] = static_cast<int>((y + _translation[1]) * _homothety[1] * _max_size);
	_cache[2] = static_cast<int>((z + _translation[2]) * _homothety[2] * _max_size);
	assert(_cache[0] >= 0 && _cache[0] <= _max_size);
	assert(_cache[1] >= 0 && _cache[1] <= _max_size);
	assert(_cache[2] >= 0 && _cache[2] <= _max_size);

	axes_to_transpose<int, 3>(_cache, _bit_size);

	// interleave the bits of transpose into one integer
	std::int64_t h = 0;
	for (std::int64_t b = 1, k = 2; k <= 2 * _bit_size; b <<= 1, k += 2) h ^= ((static_cast<std::int64_t>(_cache[0]) & b) << k);
	for (std::int64_t b = 1, k = 1; k < 2 * _bit_size; b <<= 1, k += 2) h ^= ((static_cast<std::int64_t>(_cache[1]) & b) << k);
	for (std::int64_t b = 1, k = 0; k < 2 * _bit_size; b <<= 1, k += 2) h ^= ((static_cast<std::int64_t>(_cache[2]) & b) << k);
	return h;
}

std::tuple<double, double, double> hilbert_curve_3d::coords(std::int64_t index) const
{
	assert(index >= 0); // the condition of <= 2^63 - 1 will always be true

	// un-interleave the bits
	_cache[0] = 0;
	for (std::int64_t b = 4, k = 2; k <= 2 * _bit_size; b <<= 3, k += 2) _cache[0] ^= ((index & b) >> k);
	_cache[1] = 0;
	for (std::int64_t b = 2, k = 1; k < 2 * _bit_size; b <<= 3, k += 2) _cache[1] ^= ((index & b) >> k);
	_cache[2] = 0;
	for (std::int64_t b = 1, k = 0; k < 2 * _bit_size; b <<= 3, k += 2) _cache[2] ^= ((index & b) >> k);

	transpose_to_axes<int, 3>(_cache, _bit_size);
	return std::make_tuple( static_cast<double>(_cache[0]) / static_cast<double>(_max_size) / _homothety[0] - _translation[0],
							static_cast<double>(_cache[1]) / static_cast<double>(_max_size) / _homothety[1] - _translation[1],
							static_cast<double>(_cache[2]) / static_cast<double>(_max_size) / _homothety[2] - _translation[2] );
}
