# Set default option values
CC = g++				# Compiler of choice
CFLAGS = -Wall -O0 `pkg-config --cflags opencv4`	# Compiler flags
CFILES = DisplayVideo.cpp		# .c file list
HFILES =				# .h Header file list
OFILES = $(patsubst %.cpp,%.o,$(CFILES))  # .o object file list
BINNAME = DisplayVideo			# executable file name
LDFLAGS = `pkg-config --libs opencv4`   # Linker flags to locate opencv

all: clean compile link

compile: DisplayVideo.cpp 
	$(CC) -c $(CFLAGS) $(CFILES) $(HFILES) -o $(OFILES)

link: DisplayVideo.o 
	$(CC) $(OFILES) $(LDFLAGS) -o $(BINNAME)

clean:
	rm -f $(OFILES) $(BINNAME)

