/* Really uses the library rather than only including it: `arm_relu_q7` clamps the
 * negative entries to zero in place, so the result is checkable without any reference
 * data. A header-only check would pass even if the archive were empty. */
#include <arm_nnfunctions.h>
#include <stdint.h>

int main(void) {
    int8_t data[4] = {-1, 2, -3, 4};
    arm_relu_q7(data, 4);
    if (data[0] != 0 || data[1] != 2 || data[2] != 0 || data[3] != 4) {
        return 1;
    }
    return 0;
}
