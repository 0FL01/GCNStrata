// Host-only HIP module API driver. No device source and no kernel recompilation.
#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
static int launches=0;
static void ck(hipError_t rc,const char* op){
  std::printf("{\"type\":\"hip_rc\",\"operation\":\"%s\",\"rc\":%d}\n",op,int(rc));std::fflush(stdout);
  if(rc!=hipSuccess)throw std::runtime_error(op);
}
#define CK(x) ck((x),#x)
struct Raw {std::array<uint16_t,3> input,output;std::array<unsigned,18> stats;};
static Raw run(hipFunction_t kernel,uint16_t bits,float scale,bool baseline){
  Raw r; r.input={0xa55a,bits,0x5aa5};r.output={0xa55a,0xdead,0x5aa5};r.stats.fill(0);r.stats[0]=0xa55aa55a;r.stats[17]=0x5aa55aa5;
  uint16_t *di=nullptr,*dout=nullptr;unsigned*ds=nullptr;
  CK(hipMalloc(reinterpret_cast<void**>(&di),sizeof(r.input)));
  CK(hipMalloc(reinterpret_cast<void**>(&dout),sizeof(r.output)));
  CK(hipMalloc(reinterpret_cast<void**>(&ds),sizeof(r.stats)));
  CK(hipMemcpy(di,r.input.data(),sizeof(r.input),hipMemcpyHostToDevice));
  CK(hipMemcpy(dout,r.output.data(),sizeof(r.output),hipMemcpyHostToDevice));
  CK(hipMemcpy(ds,r.stats.data(),sizeof(r.stats),hipMemcpyHostToDevice));
  uint16_t*in=di+1;uint16_t*out=dout+1;unsigned*stats=ds+1;int64_t n=1;
  void*base_args[]={&in,&out,&n,&stats,&scale};void*candidate_args[]={&in,&out,&n,&stats};
  void**args=baseline?base_args:candidate_args;
  if(++launches>28)throw std::runtime_error("launch budget");
  CK(hipModuleLaunchKernel(kernel,1,1,1,256,1,1,0,nullptr,args,nullptr));
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(r.input.data(),di,sizeof(r.input),hipMemcpyDeviceToHost));
  CK(hipMemcpy(r.output.data(),dout,sizeof(r.output),hipMemcpyDeviceToHost));
  CK(hipMemcpy(r.stats.data(),ds,sizeof(r.stats),hipMemcpyDeviceToHost));
  std::printf("{\"type\":\"raw\",\"arm\":\"%s\",\"scale\":%.0f,\"bf16\":%u,\"input\":[%u,%u,%u],\"output\":[%u,%u,%u],\"stats_with_guards\":[",baseline?"baseline":"candidate",scale,unsigned(bits),unsigned(r.input[0]),unsigned(r.input[1]),unsigned(r.input[2]),unsigned(r.output[0]),unsigned(r.output[1]),unsigned(r.output[2]));
  for(size_t i=0;i<r.stats.size();++i)std::printf("%s%u",i?",":"",r.stats[i]);std::puts("]}");std::fflush(stdout);
  CK(hipFree(di));CK(hipFree(dout));CK(hipFree(ds));return r;
}
static bool guards(const Raw&r,uint16_t bits){return r.input==std::array<uint16_t,3>{0xa55a,bits,0x5aa5} && r.output[0]==0xa55a && r.output[2]==0x5aa5 && r.stats[0]==0xa55aa55a && r.stats[17]==0x5aa55aa5;}
int main(int argc,char**argv){
  setvbuf(stdout,nullptr,_IOLBF,0);
  try{
    if(argc!=3)throw std::runtime_error("usage: sanity baseline.elf candidate.elf");
    CK(hipInit(0));CK(hipSetDevice(0));int dev=-1;CK(hipGetDevice(&dev));if(dev!=0)throw std::runtime_error("GPU0 only");
    hipDeviceProp_t p{};CK(hipGetDeviceProperties(&p,0));if(std::strncmp(p.gcnArchName,"gfx906",6))throw std::runtime_error("gfx906 required");
    char pci[32]{};CK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));if(std::strcmp(pci,"0000:05:00.0"))throw std::runtime_error("GPU0 physical PCI identity mismatch");std::printf("{\"type\":\"device\",\"ordinal\":0,\"pci\":\"%s\",\"arch\":\"%s\"}\n",pci,p.gcnArchName);
    hipModule_t base{},cand{};CK(hipModuleLoad(&base,argv[1]));CK(hipModuleLoad(&cand,argv[2]));
    hipFunction_t b{},c[2]{};CK(hipModuleGetFunction(&b,base,"_ZN6strata7prefill12_GLOBAL__N_120hc_plain_bf16_to_f16ILb0ELb1ELb1EEEvPKtP6__halflPjf"));
    CK(hipModuleGetFunction(&c[0],cand,"hc_exact_scale4096"));CK(hipModuleGetFunction(&c[1],cand,"hc_exact_scale512"));
    constexpr uint16_t patterns[]={0x0000,0x8000,0x3f80,0x0001,0x7f80,0x7fc0,0x7f81};constexpr float scales[]={4096.f,512.f};int pairs=0;
    for(int s=0;s<2;++s)for(uint16_t bits:patterns){
      Raw a=run(b,bits,scales[s],true),z=run(c[s],bits,scales[s],false);
      bool equal=a.input==z.input&&a.output==z.output&&a.stats==z.stats&&guards(a,bits)&&guards(z,bits);
      std::printf("{\"type\":\"pair\",\"index\":%d,\"equal\":%s}\n",++pairs,equal?"true":"false");std::fflush(stdout);
      if(!equal)throw std::runtime_error("exact raw parity or guard mismatch");
    }
    CK(hipModuleUnload(cand));CK(hipModuleUnload(base));
    if(pairs!=14||launches!=28)throw std::runtime_error("coverage");
    std::puts("{\"type\":\"complete\",\"pairs\":14,\"launches\":28,\"passed\":true}");return 0;
  }catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s; launches=%d\n",e.what(),launches);return 1;}
}

