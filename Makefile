CXX = g++
LD = g++

INC = -I./include -I./include/blas -I./src/include -I$(HOME)/opt/gnu-7.3.0/boost/include -I/usr/include/mkl
CXXFLAGS = -O3 -std=c++14 -mavx2 -m64 -mfma -fopenmp -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized -fpermissive $(INC)
LDFLAGS = -O2 -L/usr/lib/x86_64-linux-gnu -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5 -lpthread -lm -ldl

#CXXFLAGS += -DBENCHMARK
CXXFLAGS += -DFP_RESCALE
#CXXFLAGS += -D_BE=11 -D_BM=52
#CXXFLAGS += -D_BE=8 -D_BM=23
#CXXFLAGS += -D_BE=8 -D_BM=7
#CXXFLAGS += -D_BE=0 -D_BM=15
CXXFLAGS += -D_BE=0 -D_BM=7
CXXFLAGS += -DTHREAD_PINNING
CXXFLAGS += -D_COLMAJOR
CXXFLAGS += -DFP_INTEGER_GEMV

all: test_fp
#all: test_general_matrix_vector
#all: test_triangular_matrix_vector
#all: test_triangular_solve
#all: test_general_matrix_vector test_triangular_matrix_vector test_triangular_solve

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

