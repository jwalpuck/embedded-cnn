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
_DEPS = matrix.h vector.h neuralNetwork.h training.h file_parser.h

# convert them to point to the right place
DEPS = $(patsubst %,$(INCDIR)/%,$(_DEPS))

# put a list of the executables here
EXECUTABLES = matrix_test matrix_test2 nn_structureTest nn_forwardPropTest gradient_test file_parserTest arithmetic_test

# put a list of all the object files here for all executables (with .o endings)
_OBJ = matrix_test.o matrix_test2.o nn_structureTest.o nn_forwardPropTest.o gradient_test.o file_parserTest.o arithmetic_test.o

# convert them to point to the right place
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

# patterns for compiling source code
# $< is the file that caused the action to occur
$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -g -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: %.C $(DEPS)
	$(CC) -g -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -g -c -o $@ $< $(CFLAGS)


all:
	for i in $(EXECUTABLES); do (make $$i) ; done


# put the specific executable rules here
matrix_test: $(ODIR)/matrix_test.o
	$(CC) -g -O0 -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

matrix_test2: $(ODIR)/matrix_test2.o
	$(CC) -g -O0 -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

nn_structureTest: $(ODIR)/nn_structureTest.o
	$(CC) -g -O0 -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

nn_forwardPropTest: $(ODIR)/nn_forwardPropTest.o
	$(CC) -g -O0 -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

gradient_test: $(ODIR)/gradient_test.o
	$(CC) -g -O0 -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

file_parserTest: $(ODIR)/file_parserTest.o
	$(CC) -g -O0 -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

arithmetic_test: $(ODIR)/arithmetic_test.o
	$(CC) -g -O0 -o $(BINDIR)/$@ $^ $(LFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
