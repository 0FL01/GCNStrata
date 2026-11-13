#include <hip/hip_runtime.h>
#include <cstring>
#include <cmath>
#include "screen_core.hpp"
#include "hip_log.hpp"
#include "watchdog.hpp"
static unsigned launches=0,samples=0;
static void ck(hipError_t rc,const char*op){emit_hip_rc(int(rc),op);if(rc!=hipSuccess)throw std::runtime_error(op);}
#define CK(x) ck((x),#x)
static hipFunction_t base{},cand[2]{};
struct Device{uint16_t*in{};uint16_t*out[2]{};uint32_t*stats[2]{};};
static void allocate(Device&v,unsigned n){CK(hipMalloc(reinterpret_cast<void**>(&v.in),(size_t(n)+2)*2));for(unsigned a=0;a<2;++a){CK(hipMalloc(reinterpret_cast<void**>(&v.out[a]),(size_t(n)+2)*2));CK(hipMalloc(reinterpret_cast<void**>(&v.stats[a]),ST*4));}}
static void release(Device&v){for(unsigned a=0;a<2;++a){CK(hipFree(v.stats[a]));CK(hipFree(v.out[a]));}CK(hipFree(v.in));}
static void upload(Device&v,Fixture&f){CK(hipMemcpy(v.in,f.input.data(),f.input.size()*2,hipMemcpyHostToDevice));for(unsigned a=0;a<2;++a){f.reset_stats(a);CK(hipMemcpy(v.out[a],f.output[a].data(),f.output[a].size()*2,hipMemcpyHostToDevice));CK(hipMemcpy(v.stats[a],f.stats[a].data(),ST*4,hipMemcpyHostToDevice));}}
static void reset_stats(Device&v,Fixture&f,unsigned arm){f.reset_stats(arm);CK(hipMemcpy(v.stats[arm],f.stats[arm].data(),ST*4,hipMemcpyHostToDevice));}
struct Prepared{
 Args args;void*fixed[4];void**argv;hipFunction_t fn;unsigned grid;
 Prepared(Device&v,const Fixture&f,unsigned arm):args(v.in+1,v.out[arm]+1,f.d.n,v.stats[arm]+1+f.d.offset,float(f.d.scale)),fixed{args.p[0],args.p[1],args.p[2],args.p[3]},argv(arm?fixed:args.p),fn(arm?cand[f.d.scale==4096?0:1]:base),grid((f.d.n+255)/256){}
 hipError_t launch(){if(++launches>TOTAL)throw std::runtime_error("launch budget exceeded");return hipModuleLaunchKernel(fn,grid,1,1,256,1,1,0,nullptr,argv,nullptr);}
};
static hipError_t submit(Device&v,const Fixture&f,unsigned arm){Prepared p(v,f,arm);return p.launch();}
static void read_stats(Device&v,Fixture&f,unsigned arm){CK(hipMemcpy(f.stats[arm].data(),v.stats[arm],ST*4,hipMemcpyDeviceToHost));std::printf("{\"type\":\"sample_stats\",\"index\":%u,\"arm\":%u,\"words\":[",samples,arm);for(unsigned j=0;j<ST;++j)std::printf("%s%u",j?",":"",f.stats[arm][j]);std::puts("]}");if(!f.stats_ok(arm))throw std::runtime_error("stats flags/canaries");}
static void read_all(Device&v,Fixture&f){CK(hipMemcpy(f.input.data(),v.in,f.input.size()*2,hipMemcpyDeviceToHost));for(unsigned a=0;a<2;++a){CK(hipMemcpy(f.output[a].data(),v.out[a],f.output[a].size()*2,hipMemcpyDeviceToHost));CK(hipMemcpy(f.stats[a].data(),v.stats[a],ST*4,hipMemcpyDeviceToHost));}}
static void preflight(Device&v,Fixture&f,const char*dir){
 std::printf("{\"type\":\"preflight_begin\",\"descriptor\":%u,\"pattern\":%u}\n",f.descriptor,f.pattern);upload(v,f);
 for(unsigned a=0;a<2;++a){CK(submit(v,f,a));CK(hipDeviceSynchronize());}
 read_all(v,f);char path[1024];for(unsigned a=0;a<2;++a){std::snprintf(path,sizeof(path),"%s/preflight-d%u-p%u-a%u.bin",dir,f.descriptor,f.pattern,a);f.save(path,a);}
 bool equal=f.check();std::printf("{\"type\":\"preflight_complete\",\"descriptor\":%u,\"pattern\":%u,\"equal\":%s}\n",f.descriptor,f.pattern,equal?"true":"false");if(!equal)throw std::runtime_error("full preflight parity or guard failure");
}
int main(int argc,char**argv){
 arm_watchdog();setvbuf(stdout,nullptr,_IOLBF,0);
 try{
  if(argc!=4)throw std::runtime_error("usage: screen baseline candidate outputdir");CK(hipInit(0));CK(hipSetDevice(0));int ordinal=-1;CK(hipGetDevice(&ordinal));if(ordinal!=0)throw std::runtime_error("GPU0 only");
  hipDeviceProp_t prop{};CK(hipGetDeviceProperties(&prop,0));if(std::strncmp(prop.gcnArchName,"gfx906",6))throw std::runtime_error("gfx906 only");char pci[32]{};CK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));if(std::strcmp(pci,"0000:05:00.0"))throw std::runtime_error("physical PCI identity");
  hipModule_t bm{},cm{};CK(hipModuleLoad(&bm,argv[1]));CK(hipModuleLoad(&cm,argv[2]));CK(hipModuleGetFunction(&base,bm,"_ZN6strata7prefill12_GLOBAL__N_120hc_plain_bf16_to_f16ILb0ELb1ELb1EEEvPKtP6__halflPjf"));CK(hipModuleGetFunction(&cand[0],cm,"hc_exact_scale4096"));CK(hipModuleGetFunction(&cand[1],cm,"hc_exact_scale512"));
  Device v{},scrub{};allocate(v,DESCS[1].n);allocate(scrub,SCRUB_N);hipEvent_t begin{},end{};CK(hipEventCreate(&begin));CK(hipEventCreate(&end));
  for(unsigned d=0;d<5;++d)for(unsigned p=0;p<3;++p){Fixture f(d,p);preflight(v,f,argv[3]);}
  Fixture scrub_fixture(5,0);preflight(scrub,scrub_fixture,argv[3]);
  if(launches!=32)throw std::runtime_error("preflight count");std::puts("{\"type\":\"all_preflights_complete\",\"launches\":32}");
  for(unsigned p=0;p<3;++p)for(unsigned regime=0;regime<2;++regime)for(unsigned d=0;d<5;++d){
   Fixture f(d,p);upload(v,f);std::printf("{\"type\":\"cell_begin\",\"descriptor\":%u,\"pattern\":%u,\"regime\":%u}\n",d,p,regime);
   for(unsigned quartet=0;quartet<3;++quartet)for(unsigned position=0;position<4;++position){unsigned arm=(position==1||position==2)?1:0;
    if(regime==0){reset_stats(v,f,arm);CK(submit(v,f,arm));}else{reset_stats(scrub,scrub_fixture,0);CK(submit(scrub,scrub_fixture,0));}
    CK(hipDeviceSynchronize());reset_stats(v,f,arm);CK(hipDeviceSynchronize());
    Prepared measured(v,f,arm);
    // No JSON, copies, argument setup or success logging between event submissions.
    hipError_t r0=hipEventRecord(begin,nullptr);if(r0!=hipSuccess)ck(r0,"timed_event_start");
    hipError_t r1=measured.launch();hipError_t r2=hipEventRecord(end,nullptr);
    ck(r0,"timed_event_start");ck(r1,"timed_kernel_launch");ck(r2,"timed_event_stop");
    CK(hipEventSynchronize(end));float ms=0;hipError_t elapsed_rc=hipEventElapsedTime(&ms,begin,end);uint32_t ms_bits=0;std::memcpy(&ms_bits,&ms,sizeof(ms));
    std::printf("{\"type\":\"event_result\",\"index\":%u,\"ms_bits\":%u,\"rc\":%d}\n",samples,ms_bits,int(elapsed_rc));ck(elapsed_rc,"hipEventElapsedTime");
    if(!(ms>0)||!std::isfinite(ms))throw std::runtime_error("invalid event duration");
    read_stats(v,f,arm);
    std::printf("{\"type\":\"sample\",\"index\":%u,\"descriptor\":%u,\"pattern\":%u,\"regime\":%u,\"quartet\":%u,\"position\":%u,\"arm\":%u,\"ms\":%.9g,\"launches\":%u}\n",samples++,d,p,regime,quartet,position,arm,double(ms),launches);
   }
   read_all(v,f);bool equal=f.check();if(!equal){char path[1024];for(unsigned a=0;a<2;++a){std::snprintf(path,sizeof(path),"%s/failure-d%u-p%u-r%u-a%u.bin",argv[3],d,p,regime,a);f.save(path,a);}}
   std::printf("{\"type\":\"cell_complete\",\"descriptor\":%u,\"pattern\":%u,\"regime\":%u,\"equal\":%s}\n",d,p,regime,equal?"true":"false");if(!equal)throw std::runtime_error("post-cell full parity/guard failure");
  }
  read_all(scrub,scrub_fixture);if(!scrub_fixture.check()){char path[1024];for(unsigned a=0;a<2;++a){std::snprintf(path,sizeof(path),"%s/failure-scrub-a%u.bin",argv[3],a);scrub_fixture.save(path,a);}throw std::runtime_error("scrub final guards/parity");}
  CK(hipEventDestroy(end));CK(hipEventDestroy(begin));release(scrub);release(v);CK(hipModuleUnload(cm));CK(hipModuleUnload(bm));
  if(launches!=TOTAL||samples!=360)throw std::runtime_error("completion counts");std::puts("{\"type\":\"complete\",\"launches\":752,\"samples\":360,\"preflight_pairs\":16,\"cells\":30,\"passed\":true}");return 0;
 }catch(const std::exception&e){std::fputs("{\"type\":\"failure\",\"error\":",stdout);json_quote(e.what());std::printf(",\"launches\":%u,\"samples\":%u}\n",launches,samples);std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
