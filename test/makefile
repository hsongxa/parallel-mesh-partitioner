# uncomment to have debug build
#DEBUG ?= 1

SRC_DIR := ../src
SRC_MSH_DIR := ./mesh
VPATH := $(SRC_DIR):$(SRC_MSH_DIR)

CC := mpic++

# this is MPICH option, for OpenMPI replace -show with -showme:compile and/or -showme:link
# and remove the filter-out function
CFLAGS := $(filter-out g++, $(shell mpic++ -show)) # for compiler other than g++, replace here

CFLAGS += -O3 -std=c++17 -Wall
ifeq ($(DEBUG),1)
  CFLAGS += -g3 -O0
endif

INCL := -I$(SRC_DIR) -I$(SRC_MSH_DIR)
SRCS := $(wildcard *.cpp) $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_MSH_DIR)/*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(notdir $(SRCS)))
EXEC := test

all: $(EXEC)
$(EXEC): $(OBJS)
	$(CC) -o $@ $(OBJS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCL) -c $< -o $@

clean:	
	rm -f $(OBJS) $(EXEC) *.o
	
.PHONY : all clean
