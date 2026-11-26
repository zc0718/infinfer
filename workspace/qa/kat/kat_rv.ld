/* Linker script for the RISC-V KAT: the image starts exactly at the entry.
 *
 * Why not `-Ttext`: QEMU's virt loader jumped to the first byte of the text
 * segment rather than to `e_entry` (measured with -d in_asm), so relying on
 * e_entry silently started the program in the middle of the code. Placing
 * `.text.start` at the base makes "entry == image start" true by construction.
 *
 * RISC-V KAT 链接脚本：镜像起点就是入口。不用 `-Ttext` 的原因：实测 QEMU virt
 * 加载器跳到了 text 段首字节而不是 `e_entry`，于是程序从代码中间开始。把
 * `.text.start` 放到基址，"入口 == 镜像起点" 就成了构造性事实。
 */
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 8M
}

SECTIONS
{
    .text.start : { *(.text.start) } > RAM
    .text       : { KEEP(*(.text.reset_handler)) KEEP(*(.text.kat_fault_handler)) *(.text*) *(.rodata*) } > RAM
    .data       : { *(.data*) } > RAM AT > RAM
    .bss        : { *(.bss*) *(COMMON) } > RAM

    /* The startup copy needs both addresses: where the initialised data is
     * stored and where it must live. On a flat RAM layout they coincide, but the
     * symbols must exist because one shared startup routine serves ARM and RV. */
    _sidata = LOADADDR(.data);
    _sdata = ADDR(.data);
    _edata = ADDR(.data) + SIZEOF(.data);
    _sbss = ADDR(.bss);
    _ebss = ADDR(.bss) + SIZEOF(.bss);
    /* Stack top matches the `li sp, 0x80200000` in the start stub. */
    _estack = 0x80200000;
    end = .;
    _end = .;
    __bss_start__ = ADDR(.bss);
    __bss_end__ = ADDR(.bss) + SIZEOF(.bss);
}
