/* QEMU semihosting smoke: a freestanding program that exits QEMU with status 0.
 * Purpose: prove toolchain -> bare ELF -> QEMU before any kernel work depends on it.
 * Lives under workspace/ (not shipped); see PLAN 3.2 for the lane this de-risks. */

static void qemu_exit(void) {
    register unsigned r0 __asm__("r0") = 0x18u;    /* SYS_EXIT */
    register unsigned r1 __asm__("r1") = 0x20026u; /* ADP_Stopped_ApplicationExit */
    __asm__ volatile("bkpt 0xAB" : : "r"(r0), "r"(r1) : "memory");
    for (;;) {
    }
}

void _start(void) { qemu_exit(); }
