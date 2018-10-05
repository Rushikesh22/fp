#!/bin/bash

export OMP_NUM_THREADS=80

for m in 1 2 3 4 5
do
    echo "use Makefile.${m}"
    make clean && make -f Makefile.${m}

    for n in 447 2048
    do
	echo "matrix size n = ${n}"

	echo "general_matrix_vector"

	for i in 1 2 3 4 5 6
	do
	    ./bin/test_general_matrix_vector.x ${n} ${n} 640 1 1 >> skylake_general_matrix_vector_n${n}.log.${m}
	done
	
	for b in 8 16 32 64 128 256
	do
	    echo "block size b = ${b}"
	    for i in 1 2 3 4 5 6
	    do
		./bin/test_general_matrix_vector.x ${n} ${n} 640 ${b} 0 >> skylake_general_matrix_vector_n${n}.log.${m}
	    done
	done

	echo "triangular_matrix_vector"

	for i in 1 2 3 4 5 6
	do
	    ./bin/test_triangular_matrix_vector.x ${n} 640 1 0 1 >> skylake_triangular_matrix_vector_n${n}.log.${m}
	done
	
	for b in 8 16 32 64 128 256
	do
	    echo "block size b = ${b}"
	    for i in 1 2 3 4 5 6
	    do
		./bin/test_triangular_matrix_vector.x ${n} 640 ${b} 0 0 >> skylake_triangular_matrix_vector_n${n}.log.${m}
	    done
	done

	echo "symmetric_triangular_matrix_vector"

	for i in 1 2 3 4 5 6
	do
	    ./bin/test_triangular_matrix_vector.x ${n} 640 1 1 1 >> skylake_symmetric_triangular_matrix_vector_n${n}.log.${m}
	done
	
	for b in 8 16 32 64 128 256
	do
	    echo "block size b = ${b}"
	    for i in 1 2 3 4 5 6
	    do
		./bin/test_triangular_matrix_vector.x ${n} 640 ${b} 1 0 >> skylake_symmetric_triangular_matrix_vector_n${n}.log.${m}
	    done
	done

	echo "triangular_solve"

	for i in 1 2 3 4 5 6
	do
	    ./bin/test_triangular_solve.x ${n} 640 1 1 >> skylake_triangular_solve_n${n}.log.${m}
	done
	
	for b in 8 16 32 64 128 256
	do
	    echo "block size b = ${b}"
	    for i in 1 2 3 4 5 6
	    do
		./bin/test_triangular_solve.x ${n} 640 ${b} 0 >> skylake_triangular_solve_n${n}.log.${m}
	    done
	done
    done
done
