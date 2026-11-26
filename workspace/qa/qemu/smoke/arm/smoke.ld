/* Linker script for the QEMU Cortex-M smoke: vectors at 0x0, image follows. */
ENTRY(reset_handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

SECTIONS
{
    /* M-profile boot: QEMU takes SP from offset 0 and PC from offset 4.
     * A two-entry table turns every fault into a lockup, which hides the real
     * verdict: the CPU reads offset 0xC for the HardFault vector and jumps to
     * whatever byte follows in .text. Fill the architectural entries so a fault
     * reaches kat_fault_handler and becomes exit code 13. */
    .vectors : {
        LONG(_estack)
        LONG(reset_handler + 1)
        LONG(kat_fault_handler + 1)  /* NMI */
        LONG(kat_fault_handler + 1)  /* HardFault */
        LONG(kat_fault_handler + 1)  /* MemManage */
        LONG(kat_fault_handler + 1)  /* BusFault */
        LONG(kat_fault_handler + 1)  /* UsageFault */
        LONG(kat_fault_handler + 1) LONG(kat_fault_handler + 1)
        LONG(kat_fault_handler + 1) LONG(kat_fault_handler + 1)
        LONG(kat_fault_handler + 1)  /* SVCall */
        LONG(kat_fault_handler + 1)  /* DebugMon */
        LONG(kat_fault_handler + 1)
        LONG(kat_fault_handler + 1)  /* PendSV */
        LONG(kat_fault_handler + 1)  /* SysTick */
    } > FLASH
    /* Where the table ended up. The image sets VTOR from this at startup: a
     * loader that only fills SP and PC leaves the vector base at its reset value
     * of 0, which is the wrong table as soon as the image is not linked at 0. */
    _svectors = ADDR(.vectors);
    /* KEEP the two entry points the vector table names: a linker-script
     * reference is not a reference for --gc-sections, so without KEEP the
     * handler is collected and the fault vector becomes 0x1 (measured:
     * "loaded new PC 0x1" followed by a Data Abort at 0xfd780000).
     * 向量表点名的两个入口必须 KEEP：对 --gc-sections 而言链接脚本里的引用
     * 不算引用，不 KEEP 就会被回收，故障向量变成 0x1（实测：日志
     * "loaded new PC 0x1"，随后在 0xfd780000 发生 Data Abort）。 */
    .text    : {
        KEEP(*(.text.reset_handler))
        KEEP(*(.text.kat_fault_handler))
        *(.text*) *(.rodata*)
    } > FLASH
    /* Runtime images need the two addresses the startup copy needs: where the
     * initialised data is stored (LMA, in FLASH) and where it must live (VMA, in
     * RAM). Without the copy `.data` holds whatever RAM happened to contain --
     * measured as a garbage kernel count on mps2-an505. */
    .data    : { *(.data*) } > RAM AT > FLASH
    .bss     : { *(.bss*) *(COMMON) } > RAM
    _sidata = LOADADDR(.data);
    _sdata = ADDR(.data);
    _edata = ADDR(.data) + SIZEOF(.data);
    _sbss = ADDR(.bss);
    _ebss = ADDR(.bss) + SIZEOF(.bss);
    _estack = ORIGIN(RAM) + LENGTH(RAM);
    /* librdimon's _sbrk anchors the heap at `end`; newlib's crt0 also looks for
       these names, so define them even though this smoke skips crt0. */
    . = ALIGN(8);
    end = .;
    _end = .;
    __bss_start__ = ADDR(.bss);
    __bss_end__ = ADDR(.bss) + SIZEOF(.bss);
}
