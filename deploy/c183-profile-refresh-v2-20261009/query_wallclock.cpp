#include <hip/hip_runtime.h>
#include <cstdio>

// Read-only runtime attributes; no kernel launches, models, or settings changes.
int main() {
    const hipError_t init = hipInit(0);
    int count = 0;
    const hipError_t count_rc = hipGetDeviceCount(&count);
    bool ok = init == hipSuccess && count_rc == hipSuccess && count == 2;
    std::printf("{\"init_rc\":%d,\"count_rc\":%d,\"device_count\":%d,\"devices\":[",
                (int)init, (int)count_rc, count);
    for (int d = 0; d < count; ++d) {
        int khz = 0;
        char pci[32] = {};
        hipDeviceProp_t props = {};
        const hipError_t attr_rc = hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, d);
        const hipError_t pci_rc = hipDeviceGetPCIBusId(pci, sizeof(pci), d);
        const hipError_t props_rc = hipGetDeviceProperties(&props, d);
        std::printf("%s{\"index\":%d,\"attribute_rc\":%d,\"wall_clock_rate_khz\":%d,"
                    "\"pci_rc\":%d,\"pci\":\"%s\",\"props_rc\":%d,\"gcn_arch\":\"%s\"}",
                    d ? "," : "", d, (int)attr_rc, khz, (int)pci_rc, pci,
                    (int)props_rc, props.gcnArchName);
        ok = ok && attr_rc == hipSuccess && pci_rc == hipSuccess && props_rc == hipSuccess && khz == 25000;
    }
    std::printf("]}\n");
    return ok ? 0 : 1;
}
