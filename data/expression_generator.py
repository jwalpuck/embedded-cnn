# Generates a file with rows containing two integers and their sums in formatting to be read by file_parser.c

from __future__ import print_function
import sys
from random import randint

f = open(sys.argv[1],'w')

for i in range (0, 10): #1000
	a = randint(0, 1000)
	b = randint(0, 1000)
	c = a + b
	line_out = "%d,%d:%d\n" % (a, b, c)
	#print(line_out, file=sys.argv[1])
	f.write(line_out)
	
f.close()