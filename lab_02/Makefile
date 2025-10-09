# Set default option values
CC = gcc				# Compiler of choice
CFLAGS = -Wall -O1			# Compiler flags
CFILES = hello_world.c			# .c file list
HFILES =				# .h Header file list
OFILES = $(patsubst %.c,%.o,$(CFILES))  # .o object file list
BINNAME = hello_world			# executable file name

all: clean compile link

compile: hello_world.c
	$(CC) -c $(CFLAGS) $(CFILES) $(HFILES) -o $(OFILES)

link: hello_world.o
	$(CC) $(LDFLAGS) $(OFILES) -o $(BINNAME)

clean:
	rm -f $(OFILES) $(BINNAME)

