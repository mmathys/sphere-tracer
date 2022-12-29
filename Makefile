UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
    CC=g++-10
else
   CC=g++
endif

CFLAGS=-O3 -lm -flto -march=skylake -Wall
DEBUGFLAGS=-g -lm -march=skylake
AVXFLAGS=-march=skylake	# todo not sure what to put here

OPTIM_SRCS = src/geometry.c src/obj_vec_spheretrace.c src/ray_vec_spheretrace.c  src/spheretrace.c
REF_SRCS = reference/spheretrace.cpp

all: test spheretrace

# Validation
test: $(OPTIM_SRCS) auxiliary/load_scene.cpp run_test.cpp 
	$(CC) $(CFLAGS) $? -o $@
spheretrace: $(OPTIM_SRCS)  auxiliary/load_scene.cpp main.cpp 
	$(CC) $(CFLAGS) $? -o $@

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ test spheretrace *.ppm *.csv
