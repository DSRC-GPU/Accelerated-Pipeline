
CC = gcc
CPPFLAGS = -I/usr/include/libxml2
CFLAGS = -DDEBUG -Wall -Wextra -pedantic -std=c99
LDFLAGS = -lxml2 -lm

NVCC = nvcc
NVCCFLAGS = -DDEBUG -arch=sm_20 -O3 -Xcompiler -Wall -Xcompiler -Wextra

TARGETS = springembedr

OBJECTS = main.c.o graph.c.o gexfparser.c.o
OBJECTS_SEQ = vector.c.o force-atlas-2.c.o
OBJECTS_PAR = vector-cuda.cu.o fa2-cuda.cu.o

%.c.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: seq

seq: $(OBJECTS) $(OBJECTS_SEQ)
	$(CC) -o $(TARGETS) $(CFLAGS) $(LDFLAGS) $^

par: $(OBJECTS) $(OBJECTS_PAR)
	$(NVCC) -o $(TARGETS) $(NVCCFLAGS) $(LDFLAGS) $^

clean:
	rm -rf *.o
	rm -rf $(TARGETS)
