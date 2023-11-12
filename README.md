# parallel-mesh-partitioner
A C++/MPI implementation of the SFC (space filling curve) based parallel mesh partitioning algorithm, first presented in the paper

**R. Borrell, J.C. Cajas, D. Mira, A. Taha, S. Koric, M. Vázquez, G. Houzeaux, "Parallel mesh partitioning based on space filling curves", *Computers & Fluids*,  Vol. 173, 2018, pp. 264-272**

and later improved in

**R. Borrell, G. Oyarzun, D. Dosimont and G. Houzeaux, "Parallel SFC-based mesh partitioning and load balancing," *2019 IEEE/ACM 10th Workshop on Latest Advances in Scalable Algorithms for Large-Scale Systems (ScalA)*, 2019, pp. 72-78, doi: 10.1109/ScalA49573.2019.00014.** 

A FORTRAN implemenation of the algorithm exists in the Alya code https://www.bsc.es/research-development/research-areas/engineering-simulations/alya-high-performance-computational and its documentation can be found at https://gitlab.bsc.es/alya/alya.

### Requirement

The code depends on a MPI installation. It is built and tested with `g++ 9.3.0` and `MPICH 3.4.2`.

### Usage

The source code only contains four header files, with the main function named `partition`. It is a function template that assumes the `mesh` is initially distributed among `p` processes and it then determines a new partition of the `mesh` to `k` parts, using the `p` processes. An overload of the function template accepts the user provided custom weights of cells. To use this code, simply include the four header files in the `/src` directory into your projects -- no installaion is needed.

They require the `mesh` class to have two public member `typedef`'s, one `coordinate_type` and the other `index_type`. The former defines the type of coordinates (usually `double` or `float`) in each dimension of the 3D space. The latter defines the type (usually `int` or `long`) used to index cells in the `mesh`. In addition, the `mesh` class should provide three public member functions, `num_local_cells`, `local_bounding_box`, and `cell_centroid`. The last two functions allow the construction of the space filling curve which maps each cell to a SFC index. The cells are identified by the local numbering 0, 1, 2, ..., `num_local_cells` - 1 (i.e., no global ID's are required for cells or vertices). For existing `mesh` implementations that do not meet thses requirements, one can easily implement an adapter class to expose these `typedef`'s and member functions.

Note that this code does not redistribute mesh cells according to the new partition, it simply computes the partition and output it to the caller. Algorithms of redistributing the mesh depend on the mesh data structure and normally reside in the mesh implementation itself. A family of distributed mesh data structures are implemented in my other repository, `parallel-meshes`, at https://github.com/hsongxa/parallel-meshes.

Example usage of the code can be found in `/test`. To build the tests, go to the folder and run **`make`**. On systems with different compilers or MPI implementations than mentioned above, change the `makefile` accordingly before running **`make`**.

