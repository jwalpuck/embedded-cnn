#include <sys/time.h> // for get_time_sec
#include <time.h>
#include "../include/my_timing.h"

// Return the time in seconds
// Note to Stephanie: it doesn't work to return a float - it must be
// a double.
double get_time_sec(void)
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
} // end get_time_sec

