
CC = g++
CPPFLAGS = -I/usr/include/libxml2 -O0
CFLAGS = -DDEBUG -Wall -Wextra -pedantic -g -ffloat-store -std=c++0x
LDFLAGS = -lm -lxml2 -lpthread #-lalglib

NVCC = nvcc
CUDA_LINK = link.o
CUDA_ARCH = -arch=sm_35
NVCCFLAGS	= -G -g -lineinfo -Xcompiler -rdynamic $(CUDA_ARCH) -dc \
            --ptxas-options=-v -prec-sqrt=true -prec-div=true \
            -ccbin=`which g++` -DDEBUG  -I$(CULA_INC_PATH)
CUDA_LDFLAGS	:= $(LDFLAGS) -lcudart -lcula_lapack -lcublas
LIBS = 
CUDA_LIBS	= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64" -L"$(CULA_LIB_PATH_64)"

TARGETS = ap

OBJECTS = main.c.o graph.c.o gexfparser.c.o timer.c.o
OBJECTS_SEQ = vector.c.o force-atlas-2.c.o util.c.o \
	smoothening.c.o pca.cpp.o vector-average.c.o break-edges.c.o \
  connected-component.c.o
OBJECTS_PAR = vector.o force-atlas-2.o cuda-timer.o cuda-stream.o util.o \
	smoothening.o pca.o test-pca.o test-util.o \
  vector-average.o break-edges.o connected-component.o

%.c.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: seq

seq: $(OBJECTS) $(OBJECTS_SEQ)
	$(CC) -o $(TARGETS) $(CFLAGS) $^ $(LDFLAGS) 

par: $(OBJECTS_PAR) $(OBJECTS)
	$(NVCC) $(CUDA_ARCH) $(OBJECTS_PAR) -dlink -o $(CUDA_LINK)
	$(CC) -rdynamic -o $(TARGETS) $(CUDA_LINK) $^ $(CUDA_LIBS) $(CUDA_LDFLAGS)
	

clean:
	rm -rf *.o
	rm -rf *.cuo
	rm -rf $(TARGETS)

