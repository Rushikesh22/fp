CXX = g++
LD = g++

MKLROOT = $(HOME)/opt/intel/compilers_and_libraries_2018.0.082/linux/mkl
INTELROOT = $(MKLROOT)/../compiler
INC = -I./include -I./include/blas -I./src/include -I$(HOME)/opt/gnu-7.3.0/boost/include -I$(MKLROOT)/include -I$(INTELROOT)/include
CXXFLAGS = -O2 -std=c++14 -mavx2 -m64 -mfma -fopenmp -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized -fpermissive $(INC)
LDFLAGS = -O2 -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -L$(INTELROOT)/lib/intel64 -liomp5 -lpthread -lm -ldl

CXXFLAGS += -DBENCHMARK
CXXFLAGS += -DFP_RESCALE
#CXXFLAGS += -D_BE=11 -D_BM=52
#CXXFLAGS += -D_BE=8 -D_BM=23
#CXXFLAGS += -D_BE=8 -D_BM=7
#CXXFLAGS += -D_BE=4 -D_BM=3
#CXXFLAGS += -D_BE=0 -D_BM=15
#CXXFLAGS += -D_BE=0 -D_BM=11
CXXFLAGS += -D_BE=0 -D_BM=7
CXXFLAGS += -DUPPER_MATRIX

#all: test_fp
#all: test_general_matrix_vector
#all: test_triangular_matrix_vector
all: test_triangular_solve

###
test_fp: bin/test_fp.x

bin/test_fp.x: obj/test_fp.o
	$(LD) $(LDFLAGS) -o $@ $^

obj/test_fp.o: src/test_fp.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

###
test_general_matrix_vector: bin/test_general_matrix_vector.x

bin/test_general_matrix_vector.x: obj/test_general_matrix_vector.o obj/general_matrix_vector_kernel.o
	$(LD) $(LDFLAGS) -o $@ $^

obj/test_general_matrix_vector.o: src/test_general_matrix_vector.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

obj/general_matrix_vector_kernel.o: src/general_matrix_vector_kernel.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

###
test_triangular_matrix_vector: bin/test_triangular_matrix_vector.x

bin/test_triangular_matrix_vector.x: obj/test_triangular_matrix_vector.o obj/triangular_matrix_vector_kernel.o
	$(LD) $(LDFLAGS) -o $@ $^

obj/test_triangular_matrix_vector.o: src/test_triangular_matrix_vector.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

obj/triangular_matrix_vector_kernel.o: src/triangular_matrix_vector_kernel.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

###
test_triangular_solve: bin/test_triangular_solve.x

bin/test_triangular_solve.x: obj/test_triangular_solve.o obj/triangular_solve_kernel.o
	$(LD) $(LDFLAGS) -o $@ $^

obj/test_triangular_solve.o: src/test_triangular_solve.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

obj/triangular_solve_kernel.o: src/triangular_solve_kernel.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

###
clean:
	rm -f *~ obj/*.o bin/*.x

