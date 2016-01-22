# set the C compiler, for C++ use g++
# use this setting on most sane machines
CC = gcc

# use this line on the engin linux machines
# CC = gcc296

# set the path to the include directory
INCDIR =../include

# set the flags for the C and C++ compiler to give lots of warnings
CFLAGS = -I$(INCDIR) -O2 -Wall -Wstrict-prototypes -Wnested-externs -Wmissing-prototypes -Wmissing-declarations
CPPFLAGS = $(CFLAGS)

# path to the object file directory
ODIR = obj

# path to the lib directory
LIBDIR =../lib

# path to the bin directory
BINDIR =../bin

# libraries to include
LIBS = -lm -limageIO
LFLAGS = -L$(LIBDIR)

# put all of the relevant include files here
_DEPS = matrix.h vector.h neuralNetwork.h training.h

# convert them to point to the right place
DEPS = $(patsubst %,$(INCDIR)/%,$(_DEPS))

# put a list of the executables here
EXECUTABLES = matrix_test nn_structureTest nn_forwardPropTest gradient_test

# put a list of all the object files here for all executables (with .o endings)
_OBJ = matrix_test.o nn_structureTest.o nn_forwardPropTest.o gradient_test.o

# convert them to point to the right place
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

# patterns for compiling source code
# $< is the file that caused the action to occur
$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: %.C $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)


all:
	for i in $(EXECUTABLES); do (make $$i) ; done


# put the specific executable rules here
matrix_test: $(ODIR)/matrix_test.o
	$(CC) -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

nn_structureTest: $(ODIR)/nn_structureTest.o
	$(CC) -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

nn_forwardPropTest: $(ODIR)/nn_forwardPropTest.o
	$(CC) -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

gradient_test: $(ODIR)/gradient_test.o
	$(CC) -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 