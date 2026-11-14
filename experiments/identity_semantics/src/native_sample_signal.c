/* Out-of-loop sampling attachment. Never linked into the SHoEbox runtime.
 * The separate controller sends thread-directed SIGUSR2. The handler writes
 * one fixed-size, nonblocking record; it touches no model memory. Linux x86-64.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

static int output_fd = -1;
static void sample_handler(int signal, siginfo_t *info, void *context) {
    int saved_errno = errno;
    (void)signal; (void)info;
    ucontext_t *u = (ucontext_t *)context;
    struct timespec now;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now);
    uint64_t record[3] = {(uint64_t)syscall(SYS_gettid),
                         (uint64_t)u->uc_mcontext.gregs[REG_RIP],
                         (uint64_t)now.tv_sec * 1000000000ULL + now.tv_nsec};
    ssize_t ignored = write(output_fd, record, sizeof(record));
    (void)ignored;
    errno = saved_errno;
}
int sample_install(int fd) {
    output_fd = fd;
    struct sigaction action = {0};
    action.sa_sigaction = sample_handler;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    return sigaction(SIGUSR2, &action, 0);
}
