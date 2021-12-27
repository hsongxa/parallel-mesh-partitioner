#ifndef HILBERT_CURVE_H
#define HILBERT_CURVE_H

#include <cassert>
#include <cinttypes>
#include <tuple>

// "Programming the Hilbert Curve", John Skilling, 2004,
// Baysian Inference and Maximum Entropy Methods in Science and Engineering:
// 23rd International Workshop, edited by G. Erickson and Y. Zhai

template<typename UINT, int DIM>
void transpose_to_axes(UINT* x, int b = sizeof(UINT) * 8)
{
	assert(b >= 1 && b <= sizeof(UINT) * 8);
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
	assert(b >= 1 && b <= sizeof(UINT) * 8);
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

// 3D hilbert curve encoded in one int_64: 21-level recursions in each dimension
class hilbert_curve_3d
{
public:
	hilbert_curve_3d(double x_min, double y_min, double z_min, double x_max, double y_max, double z_max)
		: _homothety{ x_min < x_max ? 1.0 / (x_max - x_min) : 1.0,
					  y_min < y_max ? 1.0 / (y_max - y_min) : 1.0,
					  z_min < z_max ? 1.0 / (z_max - z_min) : 1.0 },
		  _translation{ -x_min, -y_min, -z_min } {}

	// hilbert index for an arbitrary point located in the defined extent
	std::int64_t index(double x, double y, double z) const;

	// the (approximate) location of the point corresponding to the given hilbert index
	std::tuple<double, double, double> coords(std::int64_t index) const;

private:
	static constexpr int _bit_size = 21;
	static constexpr int _max_size = 2097151; // 2^21 - 1 = 2097151
	const double _homothety[3];
	const double _translation[3];
	mutable int	 _cache[3];
};

#endif
