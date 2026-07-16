#include "affinity.h"


static pid_t gettid_() { 
    return (pid_t)syscall(SYS_gettid);
}

static bool pin_tid(pid_t tid, std::initializer_list<int> cpus) {
  cpu_set_t cs; CPU_ZERO(&cs);
  for (int c : cpus) {
    if (c < 0 || c >= CPU_SETSIZE) {
      return false;
    }
    CPU_SET(c, &cs);
  }
  return sched_setaffinity(tid, sizeof(cs), &cs) == 0;
}

bool pin_current(std::initializer_list<int> cpus) {
  return pin_tid(gettid_(), cpus);
}
