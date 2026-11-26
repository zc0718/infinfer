/* QEMU smoke for RISC-V: exit through the virt machine's SiFive test finisher.
 * Writing FINISHER_PASS (0x5555) to 0x100000 makes QEMU exit with status 0.
 * Semihosting was tried first and is not serviced on QEMU 8.2 RISC-V; the
 * finisher is the standard bare-metal test exit and needs no ABI assumptions. */

static void qemu_exit_pass(void) {
    volatile unsigned* const finisher = (volatile unsigned*)0x100000u;
    *finisher = 0x5555u; /* FINISHER_PASS */
    for (;;) {
    }
}

void _start(void) { qemu_exit_pass(); }
