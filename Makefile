all: gpu mpi sse

# GPU
gpu: gputest.cu
	nvcc gputest.cu -o gputest

# MPI
mpi: mpitest.c
	mpic++ mpitest.c -o mpitest -lm

# SSE
sse: ssetest.c
	gcc ssetest.c -w -o ssetest -lm -fopenmp


clean:
	rm gputest mpitest ssetest 
