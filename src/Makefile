
CC = g++
CPPFLAGS = -I/usr/include/libxml2
CFLAGS = -DDEBUG -Wall -Wextra -pedantic
LDFLAGS = -lxml2 -lm

NVCC = nvcc
NVCCFLAGS = -DDEBUG -arch=compute_35 -code=sm_35 --ptxas-options=-v -dc
NVCCLDFLAGS = -dlink -arch=compute_35 -code=sm_35

TARGETS = springembedr

OBJECTS = main.c.o graph.c.o gexfparser.c.o
OBJECTS_SEQ = vector.c.o force-atlas-2.c.o
OBJECTS_PAR = fa2-cuda.cuo vector-cuda.cuo 

%.c.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

%.cuo: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: seq

seq: $(OBJECTS) $(OBJECTS_SEQ)
	$(CC) -o $(TARGETS) $(CFLAGS) $(LDFLAGS) $^

par: $(OBJECTS_PAR) $(OBJECTS)
	$(NVCC) -o $(TARGETS) $(NVCCLDFLAGS) $^


clean:
	rm -rf *.o
	rm -rf *.cuo
	rm -rf $(TARGETS)

