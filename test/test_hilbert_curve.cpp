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

#include "hilbert_curve.h"

int test_hilbert_curve ()
{
  pmp::hilbert_curve_3d c1(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);

  auto index = c1.index(0.0, 0.0, 0.0, 1);
  std::tuple<double, double, double> coords = c1.coords(0, 1);
  std::cout << "hilbert index = " << index << ", coords are (";
  std::cout << std::get<0>(coords) << ", " << std::get<1>(coords) << ", " << std::get<2>(coords) << ")" << std::endl;

  index = c1.index(1.0, 0.0, 0.0, 1);
  coords = c1.coords(7, 1);
  std::cout << "hilbert index = " << index << ", coords are (";
  std::cout << std::get<0>(coords) << ", " << std::get<1>(coords) << ", " << std::get<2>(coords) << ")" << std::endl;

  index = c1.index(1.0, 1.0, 0.0, 1);
  coords = c1.coords(4, 1);
  std::cout << "hilbert index = " << index << ", coords are (";
  std::cout << std::get<0>(coords) << ", " << std::get<1>(coords) << ", " << std::get<2>(coords) << ")" << std::endl;

  index = c1.index(0.0, 1.0, 0.0, 1);
  coords = c1.coords(3, 1);
  std::cout << "hilbert index = " << index << ", coords are (";
  std::cout << std::get<0>(coords) << ", " << std::get<1>(coords) << ", " << std::get<2>(coords) << ")" << std::endl;

  index = c1.index(0.0, 0.0, 1.0, 1);
  coords = c1.coords(1, 1);
  std::cout << "hilbert index = " << index << ", coords are (";
  std::cout << std::get<0>(coords) << ", " << std::get<1>(coords) << ", " << std::get<2>(coords) << ")" << std::endl;

  index = c1.index(1.0, 0.0, 1.0, 1);
  coords = c1.coords(6, 1);
  std::cout << "hilbert index = " << index << ", coords are (";
  std::cout << std::get<0>(coords) << ", " << std::get<1>(coords) << ", " << std::get<2>(coords) << ")" << std::endl;

  index = c1.index(1.0, 1.0, 1.0, 1);
  coords = c1.coords(5, 1);
  std::cout << "hilbert index = " << index << ", coords are (";
  std::cout << std::get<0>(coords) << ", " << std::get<1>(coords) << ", " << std::get<2>(coords) << ")" << std::endl;

  index = c1.index(0.0, 1.0, 1.0, 1);
  coords = c1.coords(2, 1);
  std::cout << "hilbert index = " << index << ", coords are (";
  std::cout << std::get<0>(coords) << ", " << std::get<1>(coords) << ", " << std::get<2>(coords) << ")" << std::endl;


  pmp::hilbert_curve_3d c2((float)0.0, (float)0.0, (float)0.0, (float)2.0, (float)2.0, (float)2.0);

  index = c2.index((float)0.5, (float)0.5, (float)0.5, 2);
  std::cout << "hilbert index = " << index << std::endl;

  index = c2.index((float)0.5, (float)0.5, (float)1.5, 2);
  std::cout << "hilbert index = " << index << std::endl;

  index = c2.index((float)0.5, (float)1.5, (float)1.5, 2);
  std::cout << "hilbert index = " << index << std::endl;

  index = c2.index((float)0.5, (float)1.5, (float)0.5, 2);
  std::cout << "hilbert index = " << index << std::endl;

  index = c2.index((float)1.5, (float)1.5, (float)0.5, 2);
  std::cout << "hilbert index = " << index << std::endl;

  index = c2.index((float)1.5, (float)1.5, (float)1.5, 2);
  std::cout << "hilbert index = " << index << std::endl;

  index = c2.index((float)1.5, (float)0.5, (float)1.5, 2);
  std::cout << "hilbert index = " << index << std::endl;

  index = c2.index((float)1.5, (float)0.5, (float)0.5, 2);
  std::cout << "hilbert index = " << index << std::endl;

  return 0;
}
