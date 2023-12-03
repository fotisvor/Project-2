SRCDIR := ../common
HEADERDIR := ../../include
OBJDIR := ../../obj

CC := g++
CFLAGS := -std=c++14 -Ofast -g3 -Wall -Wextra -I$(HEADERDIR)

BINARY_LSH := lsh
BINARY_MRNG := mrngR

SOURCES_LSH := $(shell find $(SRCDIR) -name '*.cpp')
SOURCES_LSH += lshmain.cpp

SOURCES_MRNG := $(shell find $(SRCDIR) -name '*.cpp')
SOURCES_MRNG += mrngR.cpp

OBJECTS_LSH := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES_LSH))
OBJECTS_MRNG := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES_MRNG))


all: $(BINARY_LSH) $(BINARY_MRNG)

$(BINARY_LSH): $(OBJECTS_LSH)
	$(CC) $(CFLAGS) $(OBJECTS_LSH) -o $(BINARY_LSH)

$(BINARY_MRNG): $(OBJECTS_MRNG)
	$(CC) $(CFLAGS) $(OBJECTS_MRNG) -o $(BINARY_MRNG)


$(OBJDIR):
	@mkdir $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf output $(BINARY_LSH) $(BINARY_MRNG) $(OBJDIR)/*
