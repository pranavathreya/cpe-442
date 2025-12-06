# Directories
IDIR     := include
SRCDIR   := src
BUILDDIR := build
BINDIR   := bin
MEDIADIR := media

# Variables
VIDFILE := $(MEDIADIR)/bbb_sunflower_2160p_60fps_normal.mp4

# Set default option values
CC = g++				# Compiler of choice
CFLAGS = -Wall -O0 `pkg-config --cflags opencv4`	# Compiler flags
CFILES = $(SRCDIR)/DisplayVideo.cpp		# .c file list
HFILES =				# .h Header file list

# Sources, objects, and auto-deps
SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OFILES := $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES)) # Object files
BINNAME = $(BINDIR)/DisplayVideo			# executable file name
LDFLAGS = `pkg-config --libs opencv4 papi`   # Linker flags to locate opencv

all: clean compile link

compile: $(SRCDIR)/DisplayVideo.cpp 
	$(CC) -c $(CFLAGS) $(CFILES) $(HFILES) -o $(OFILES)

link: $(BUILDDIR)/DisplayVideo.o 
	$(CC) $(OFILES) $(LDFLAGS) -o $(BINNAME)

clean:
	rm -f $(OFILES) $(BINNAME)

run: $(BINNAME)
	./$(BINNAME) $(VIDFILE) 
