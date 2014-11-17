
CC = g++
CPPFLAGS = -I/usr/include/libxml2 -I../alglib/cpp/src
CFLAGS = -Wall -Wextra -pedantic -ffloat-store -std=c++0x
LDFLAGS = -lm -lxml2 -lpthread -L../alglib/cpp/lib -lalglib

NVCC = nvcc
CUDA_LINK = link.o
CUDA_ARCH = -arch=sm_20
NVCCFLAGS	= -lineinfo -Xcompiler -rdynamic $(CUDA_ARCH) -dc \
            --ptxas-options=-v -prec-sqrt=true -prec-div=true \
            -ccbin=`which g++`  -I$(CULA_INC_PATH)
CUDA_LDFLAGS	:= $(LDFLAGS) -lcudart -lcula_lapack -lcublas
LIBS = 
CUDA_LIBS	= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64" -L"$(CULA_LIB_PATH_64)"


TARGETS = ap

MAIN = main.c.o
SEP_MAIN = se-performance.c.o
OBJECTS = graph.c.o gexfparser.c.o pipeline.c.o
OBJECTS_SEQ = vector.c.o force-atlas-2.c.o util.c.o \
	smoothening.c.o pca.cpp.o vector-average.c.o break-edges.c.o \
  connected-component.c.o pipeline.c.o timer.c.o
OBJECTS_PAR = vector.o force-atlas-2.o cuda-stream.o util.o \
	smoothening.o pca.o test-pca.o test-util.o \
  vector-average.o break-edges.o connected-component.o timer.o

%.c.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: seq

seq-debug: CFLAGS += -DDEBUG -g
seq-debug: seq

seq: $(MAIN) $(OBJECTS) $(OBJECTS_SEQ)
	$(CC) -o $(TARGETS) $(CFLAGS) $^ $(LDFLAGS) 

par-debug: NVCCFLAGS += -DDEBUG -G -g
par-debug: CFLAGS += -DDEBUG -g
par-debug: par

par: $(MAIN) $(OBJECTS_PAR) $(OBJECTS)
	$(NVCC) $(CUDA_ARCH) $(OBJECTS_PAR) -dlink -o $(CUDA_LINK)
	$(CC) -rdynamic -o $(TARGETS) $(CUDA_LINK) $^ $(CUDA_LIBS) $(CUDA_LDFLAGS)

sep: $(SEP_MAIN) $(OBJECTS_PAR) $(OBJECTS)
	$(NVCC) $(CUDA_ARCH) $(OBJECTS_PAR) -dlink -o $(CUDA_LINK)
	$(CC) -rdynamic -o $(TARGETS) $(CUDA_LINK) $^ $(CUDA_LIBS) $(CUDA_LDFLAGS)

clean:
	rm -rf *.o
	rm -rf *.cuo
	rm -rf $(TARGETS)

