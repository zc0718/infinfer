/* QEMU newlib smoke: link a real libc function (snprintf) from newlib and print
 * the result through ARM semihosting. This is the shape an M-profile KAT will
 * use, so proof now avoids discovering a newlib problem inside the KAT wiring.
 * Lives under workspace/ (not shipped). */

#include <stdio.h>

extern void initialise_monitor_handles(void); /* librdimon: semihosting-backed syscalls */

static void sh_exit(void) {
    register unsigned r0 __asm__("r0") = 0x18u;    /* SYS_EXIT */
    register unsigned r1 __asm__("r1") = 0x20026u; /* ADP_Stopped_ApplicationExit */
    __asm__ volatile("bkpt 0xAB" : : "r"(r0), "r"(r1) : "memory");
    for (;;) {
    }
}

void reset_handler(void) {
    initialise_monitor_handles();
    printf("newlib-ok %d\n", 42);
    sh_exit();
}
