#include <iostream>
#include <linux/types.h>
#include <pthread.h>

static_assert(sizeof(__u32) == 4);
static_assert(sizeof(pthread_t) > 0);
