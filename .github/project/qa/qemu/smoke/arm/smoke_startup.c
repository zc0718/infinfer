/* QEMU semihosting smoke for Cortex-M: vector table + reset handler.
 * QEMU loads a raw -kernel image by taking SP from offset 0 and PC from offset 4,
 * which is the same shape a real KAT image has -- proving this now avoids a
 * rework when kernels arrive. Lives under workspace/ (not shipped). */

extern unsigned _estack;

static void qemu_exit(void) {
    register unsigned r0 __asm__("r0") = 0x18u;    /* SYS_EXIT */
    register unsigned r1 __asm__("r1") = 0x20026u; /* ADP_Stopped_ApplicationExit */
    __asm__ volatile("bkpt 0xAB" : : "r"(r0), "r"(r1) : "memory");
    for (;;) {
    }
}

void reset_handler(void) { qemu_exit(); }
