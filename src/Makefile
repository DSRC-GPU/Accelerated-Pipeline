
CC=gcc
CPPFLAGS=-I/usr/include/libxml2
CFLAGS=-Wall -Wextra -pedantic -std=c99
LDFLAGS=-lxml2 -lm

NVCC = nvcc
NVCCFLAGS = -DDEBUG -Wall -Wextra

TARGETS = springembedr

OBJECTS = main.c.o graph.c.o gexfparser.c.o vector.c.o
OBJECTS_SEQ = force-atlas-2.c.o
OBJECTS_PAR = fa2-cuda.cu.o

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
