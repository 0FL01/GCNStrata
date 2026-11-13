#pragma once
#include <csignal>
#include <unistd.h>
inline void arm_watchdog(unsigned seconds=300){std::signal(SIGALRM,SIG_DFL);alarm(seconds);}
