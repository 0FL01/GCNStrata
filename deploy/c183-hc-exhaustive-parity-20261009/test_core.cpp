#include "core.hpp"
#include "hip_log.hpp"
#include "watchdog.hpp"
#include <cassert>
#include <cstring>
int main(int argc,char**argv){
 if(argc==2 && std::strcmp(argv[1],"--watchdog")==0){arm_watchdog(1);for(;;)pause();}
 assert(argc==2);assert(sizeof(void*)==8&&sizeof(int64_t)==8&&sizeof(uint32_t)==4&&sizeof(uint16_t)==2);
 uint16_t in[4]{},out[4]{};uint32_t stats[23]{};
 Args a(in+1,out+1,INT64_C(0x123456789),stats+6,4096.f);
 assert(*static_cast<uint16_t**>(a.pointers[0])==in+1);assert(*static_cast<uint16_t**>(a.pointers[1])==out+1);
 assert(*static_cast<int64_t*>(a.pointers[2])==INT64_C(0x123456789));assert(*static_cast<uint32_t**>(a.pointers[3])==stats+6);assert(*static_cast<float*>(a.pointers[4])==4096.f);
 emit_hip_rc(7,"quote\" slash\\ tab\t newline\n carriage\r \001");
 Batch v(0,4096,32768,2,1,0);assert(v.input[1]==0x8000&&v.input[4]==0x8000&&v.input[7]==0x8001);
 v.save(std::string(argv[1])+"/fixture.bin");
 Batch tail(1,512,0,1,448,5);assert(tail.stats[5]==GUARD&&tail.stats[6]==0&&tail.stats[21]==0&&tail.stats[22]==GUARD);tail.save(std::string(argv[1])+"/tail.bin");
}
