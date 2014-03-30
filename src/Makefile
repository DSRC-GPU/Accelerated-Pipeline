
CC = g++
CPPFLAGS = -I/usr/include/libxml2
CFLAGS = -DDEBUG -Wall -Wextra -pedantic
LDFLAGS = -lm -lxml2 -lpthread

NVCC = nvcc
CUDA_LINK = link.o
CUDA_ARCH = -arch=sm_20
NVCCFLAGS	= $(CUDA_ARCH) -dc --ptxas-options=-v
CUDA_LDFLAGS	:= $(LDFLAGS) -lcudart
LIBS = 
CUDA_LIBS	= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64"

TARGETS = springembedr

OBJECTS = main.c.o graph.c.o gexfparser.c.o
OBJECTS_SEQ = vector.c.o force-atlas-2.c.o
OBJECTS_PAR = vector.o force-atlas-2.o

%.c.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: seq

seq: $(OBJECTS) $(OBJECTS_SEQ)
	$(CC) -o $(TARGETS) $(CFLAGS) $(LDFLAGS) $^

par: $(OBJECTS_PAR) $(OBJECTS)
	$(NVCC) $(CUDA_ARCH) $(OBJECTS_PAR) -dlink -o $(CUDA_LINK)
	$(CC) -o $(TARGETS) $(CUDA_LINK) $^ $(CUDA_LIBS) $(CUDA_LDFLAGS)
	

clean:
	rm -rf *.o
	rm -rf *.cuo
	rm -rf $(TARGETS)

