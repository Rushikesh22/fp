CXX = g++
LD = g++

MKLROOT = $(HOME)/opt/intel/compilers_and_libraries_2018.0.082/linux/mkl
INTELROOT = $(MKLROOT)/../compiler
INC = -I./include -I./include/blas -I./src/include -I$(HOME)/opt/gnu-7.3.0/boost/include -I$(MKLROOT)/include -I$(INTELROOT)/include
CXXFLAGS = -O3 -std=c++14 -mavx2 -m64 -mfma -fopenmp -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized $(INC)
LDFLAGS = -O2 -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -L$(INTELROOT)/lib/intel64 -liomp5 -lpthread -lm -ldl

#CXXFLAGS += -DFULL_MATRIX
CXXFLAGS += -DFP_RESCALE
#CXXFLAGS += -D_BE=8 -D_BM=23
#CXXFLAGS += -D_BE=8 -D_BM=7
#CXXFLAGS += -D_BE=11 -D_BM=4
CXXFLAGS += -D_BE=0 -D_BM=16
#CXXFLAGS += -D_BE=0 -D_BM=12
#CXXFLAGS += -D_BE=5 -D_BM=6
#CXXFLAGS += -D_BE=0 -D_BM=8
#CXXFLAGS += -D_BE=5 -D_BM=2

target: bin/matrix_vector.x

bin/matrix_vector.x: obj/matrix_vector.o obj/matrix_vector_kernel.o
	$(LD) $(LDFLAGS) -o $@ $^

obj/matrix_vector.o: src/matrix_vector.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

obj/matrix_vector_kernel.o: src/matrix_vector_kernel.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -f *~ obj/*.o bin/matrix_vector.x

