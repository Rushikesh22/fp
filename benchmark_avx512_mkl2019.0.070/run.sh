#!/bin/bash

mkdir -p bin

omp_threads=80
export OMP_NUM_THREADS=${omp_threads}

for m in 64BIT 32BIT 16BIT_FLOAT 16BIT_FIXED 8BIT_FIXED
do
    work_dir=`pwd`
    echo "#######################################"
    echo "### version: ${m}"
    echo "#######################################"
    cp Makefile.avx512.${m} ..
    cd ..
    make clean && make -f Makefile.avx512.${m}
    mv bin/*.x ${work_dir}/bin
    rm Makefile.avx512.${m}
    cd ${work_dir}

    for n in 447 2048
    do
	echo "matrix size n = ${n}"

	if [ "${n}" == "447" ]
	then
	    N=6400
	else
	    N=640
	fi

	echo "general_matrix_vector"

	for i in 1 2 3 4 5 6
	do
	    echo "reference: n${n} N${N} i${i}"
	    ./bin/test_general_matrix_vector.x ${n} ${n} ${N} 1 1 > skylake_general_matrix_vector_${m}_n${n}_threads${omp_threads}_${i}.log
	    cat skylake_general_matrix_vector_${m}_n${n}_threads${omp_threads}_${i}.log | grep gflops
	done
	
	for b in 8 16 32 64 128 256
	do
	    for i in 1 2 3 4 5 6
	    do
		echo "own: n${n} N${N} bs${b} i${i}"
		./bin/test_general_matrix_vector.x ${n} ${n} ${N} ${b} 0 > skylake_general_matrix_vector_${m}_n${n}_bs${b}_threads${omp_threads}_${i}.log
		cat skylake_general_matrix_vector_${m}_n${n}_bs${b}_threads${omp_threads}_${i}.log | grep gflops
	    done
	done

	echo "triangular_matrix_vector"

	for i in 1 2 3 4 5 6
	do
	    echo "reference: n${n} N${N} i${i}"
	    ./bin/test_triangular_matrix_vector.x ${n} ${N} 1 0 1 > skylake_triangular_matrix_vector_${m}_n${n}_threads${omp_threads}_${i}.log
	    cat skylake_triangular_matrix_vector_${m}_n${n}_threads${omp_threads}_${i}.log | grep gflops
	done
	
	for b in 8 16 32 64 128 256
	do
	    for i in 1 2 3 4 5 6
	    do
		echo "own: n${n} N${N} bs${b} i${i}"
		./bin/test_triangular_matrix_vector.x ${n} ${N} ${b} 0 0 > skylake_triangular_matrix_vector_${m}_n${n}_bs${b}_threads${omp_threads}_${i}.log
		cat skylake_triangular_matrix_vector_${m}_n${n}_bs${b}_threads${omp_threads}_${i}.log | grep gflops
	    done
	done

	echo "symmetric_triangular_matrix_vector"

	for i in 1 2 3 4 5 6
	do
	    echo "reference: n${n} N${N} i${i}"
	    ./bin/test_triangular_matrix_vector.x ${n} ${N} 1 1 1 > skylake_symmetric_triangular_matrix_vector_${m}_n${n}_threads${omp_threads}_${i}.log
	    cat skylake_symmetric_triangular_matrix_vector_${m}_n${n}_threads${omp_threads}_${i}.log | grep gflops
	done
	
	for b in 8 16 32 64 128 256
	do
	    for i in 1 2 3 4 5 6
	    do
		echo "own: n${n} N${N} bs${b} i${i}"
		./bin/test_triangular_matrix_vector.x ${n} ${N} ${b} 1 0 > skylake_symmetric_triangular_matrix_vector_${m}_n${n}_bs${b}_threads${omp_threads}_${i}.log
		cat skylake_symmetric_triangular_matrix_vector_${m}_n${n}_bs${b}_threads${omp_threads}_${i}.log | grep gflops
	    done
	done

	echo "triangular_solve"

	for i in 1 2 3 4 5 6
	do
	    echo "reference: n${n} N${N} i${i}"
	    ./bin/test_triangular_solve.x ${n} ${N} 1 1 > skylake_triangular_solve_${m}_n${n}_threads${omp_threads}_${i}.log
	    cat skylake_triangular_solve_${m}_n${n}_threads${omp_threads}_${i}.log | grep gflops
	done
	
	for b in 8 16 32 64 128 256
	do
	    for i in 1 2 3 4 5 6
	    do
		echo "own: n${n} N${N} bs${b} i${i}"
		./bin/test_triangular_solve.x ${n} ${N} ${b} 0 > skylake_triangular_solve_${m}_n${n}_bs${b}_threads${omp_threads}_${i}.log
		cat skylake_triangular_solve_${m}_n${n}_bs${b}_threads${omp_threads}_${i}.log | grep gflops
	    done
	done
    done
done
