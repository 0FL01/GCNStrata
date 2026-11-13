#include "screen_core.hpp"
#include "hip_log.hpp"
#include "watchdog.hpp"
#include <cassert>
#include <cstring>
int main(int argc,char**argv){
 assert(argc==2);if(std::strcmp(argv[1],"--watchdog")==0){arm_watchdog(1);for(;;)pause();}
 uint16_t in[2]{},out[2]{};uint32_t st[23]{};Args a(in,out,INT64_C(0x123456789),st+6,512.f);
 assert(*static_cast<uint16_t**>(a.p[0])==in&&*static_cast<uint16_t**>(a.p[1])==out);assert(*static_cast<int64_t*>(a.p[2])==INT64_C(0x123456789));assert(*static_cast<uint32_t**>(a.p[3])==st+6&&*static_cast<float*>(a.p[4])==512.f);
 emit_hip_rc(7,"quote\" slash\\ tab\t newline\n \001");
 for(unsigned pattern=0;pattern<3;++pattern){Fixture f(4,pattern);for(unsigned arm=0;arm<2;++arm){if(pattern)for(unsigned j=2+f.d.offset;j<=4+f.d.offset;++j)f.stats[arm][j]=1;}assert(f.check());
  for(unsigned arm=0;arm<2;++arm)f.save(std::string(argv[1])+"/preflight-d4-p"+std::to_string(pattern)+"-a"+std::to_string(arm)+".bin",arm);
  f.stats[0][1+f.d.offset]=1;assert(!f.check());
 }
}
