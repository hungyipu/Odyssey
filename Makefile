#before run the Makefile: module load cuda/8.0.44
CUDA_PATH = /usr/local/cuda
SRC_PATH  = ./src
CUDA      = nvcc
#CUDAFLAGS = -arch=compute_20 -c
CUDAFLAGS = -c 
#-Wno-deprecated-gpu-targets
CPP       = g++
CFLAGS    = -c -g -lm -Wall

all: cpp cu exe

cpp:
	@${CPP} ${CFLAGS} -I. -I${CUDA_PATH}/include ${SRC_PATH}/main.cpp  -o main.cpp.o
	@${CPP} ${CFLAGS} -I. -I${CUDA_PATH}/include ${SRC_PATH}/task1.cpp -o task1.cpp.o
	@${CPP} ${CFLAGS} -I. -I${CUDA_PATH}/include ${SRC_PATH}/task2.cpp -o task2.cpp.o  
cu:

	@${CUDA} ${CUDAFLAGS} -I. -I${CUDA_PATH}/include ${SRC_PATH}/Odyssey.cu -o Odyssey.cu.o

exe:

	@${CPP} -o exec *.o -L${CUDA_PATH}/lib64 -lcudart
	$(info =================================)
	$(info exe file is ready to run: ./exec)
	$(info =================================)
	
plot:
	@gnuplot < plot/plot_task2.gp	

clean:
	@rm -f *.o

clean_all:
	@rm -f *.o
	@rm -f exec
	@rm -f *.png
