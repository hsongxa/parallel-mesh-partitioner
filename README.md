# parallel-mesh-partitioner
A C++/MPI implementation of the SFC (space filling curve) based parallel mesh partitioning algorithm, first presented in the paper

**R. Borrell, J.C. Cajas, D. Mira, A. Taha, S. Koric, M. Vázquez, G. Houzeaux, "Parallel mesh partitioning based on space filling curves", *Computers & Fluids*,  Volume 173, 2018,Pages 264-272**

and later improved in

**R. Borrell, G. Oyarzun, D. Dosimont and G. Houzeaux, "Parallel SFC-based mesh partitioning and load balancing," *2019 IEEE/ACM 10th Workshop on Latest Advances in Scalable Algorithms for Large-Scale Systems (ScalA)*, 2019, pp. 72-78, doi: 10.1109/ScalA49573.2019.00014.** 

A FORTRAN implemenation of the algorithm exists in the Alya code https://www.bsc.es/research-development/research-areas/engineering-simulations/alya-high-performance-computational and its documentation can be found at https://gitlab.bsc.es/alya/alya.

### Requirement

The code is built and tested with `g++ 9.3.0` and `MPICH 3.4.2`.

### Usage

Example usage of the code can be found in `/test`. To run the tests, go to the folder and run **`make`**. The path to the MPI installation is assumed to be `/usr/local/bin/`. For a different path, change in the **`makefile`** accordingly before running **`make`**.




