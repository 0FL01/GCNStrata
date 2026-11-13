// Host-only module API. No device kernels or performance measurements.
#include <hip/hip_runtime.h>
#include <cstring>
#include "watchdog.hpp"
#include "core.hpp"
#include "hip_log.hpp"
static unsigned launches=0,batches=0;
static void ck(hipError_t rc,const char*op){emit_hip_rc(int(rc),op);if(rc!=hipSuccess)throw std::runtime_error(op);}
#define CK(x) ck((x),#x)
static uint16_t*di=nullptr,*doo=nullptr;static uint32_t*ds=nullptr;
static void run(Batch&v,hipFunction_t base,hipFunction_t cand,const char*dir){
 char stem[1024];std::snprintf(stem,sizeof(stem),"%s/batch-%04u",dir,batches);
 std::string path(stem);v.save(path+".initial.bin");
 std::printf("{\"type\":\"batch_begin\",\"batch\":%u,\"kind\":%u,\"scale\":%u,\"start\":%u,\"pairs\":%u,\"n\":%u,\"offset\":%u,\"launches_before\":%u}\n",batches,v.kind,v.scale,v.start,v.pairs,v.n,v.offset,launches);
 CK(hipMemcpy(di,v.input.data(),v.input.size()*2,hipMemcpyHostToDevice));
 CK(hipMemcpy(doo,v.output.data(),v.output.size()*2,hipMemcpyHostToDevice));
 CK(hipMemcpy(ds,v.stats.data(),v.stats.size()*4,hipMemcpyHostToDevice));
 for(unsigned r=0;r<v.pairs*2;++r){
  Args a(di+r*(v.n+2)+1,doo+r*(v.n+2)+1,v.n,ds+r*ST+1+v.offset,float(v.scale));
  if(++launches>TOTAL_LAUNCHES)throw std::runtime_error("launch budget");
  void* fixed_args[]={a.pointers[0],a.pointers[1],a.pointers[2],a.pointers[3]};
  CK(hipModuleLaunchKernel(r%2?cand:base,(v.n+255)/256 ? (v.n+255)/256:1,1,1,256,1,1,0,nullptr,r%2?fixed_args:a.pointers,nullptr));
 }
 CK(hipDeviceSynchronize());
 CK(hipMemcpy(v.input.data(),di,v.input.size()*2,hipMemcpyDeviceToHost));
 v.save(path+".input-copied.bin");
 CK(hipMemcpy(v.output.data(),doo,v.output.size()*2,hipMemcpyDeviceToHost));
 v.save(path+".output-copied.bin");
 CK(hipMemcpy(v.stats.data(),ds,v.stats.size()*4,hipMemcpyDeviceToHost));
 v.save(path+".complete.bin");
 Batch expected(v.kind,v.scale,v.start,v.pairs,v.n,v.offset);bool equal=v.input==expected.input;
 for(unsigned r=0;r<v.pairs*2;++r){
  unsigned at=r*(v.n+2);equal &= v.output[at]==0xa55a && v.output[at+v.n+1]==0x5aa5;
  for(unsigned j=0;j<ST;++j)if(j<1+v.offset||j>=17+v.offset)equal &= v.stats[r*ST+j]==GUARD;
  for(unsigned j=0;j<16;++j){uint32_t flag=v.stats[r*ST+1+v.offset+j];equal &= (v.n && j>=1 && j<=3)?flag<=1:flag==0;}
  if(r%2){for(unsigned j=0;j<v.n+2;++j)equal &= v.output[at+j]==v.output[at-(v.n+2)+j];for(unsigned j=0;j<ST;++j)equal &=v.stats[r*ST+j]==v.stats[(r-1)*ST+j];}
 }
 std::printf("{\"type\":\"batch_complete\",\"batch\":%u,\"launches\":%u,\"equal\":%s}\n",batches,launches,equal?"true":"false");++batches;
 if(!equal)throw std::runtime_error("raw parity/input/guard mismatch");
}
int main(int argc,char**argv){
 arm_watchdog(); // Hard 300-second process watchdog, including initialization.
 setvbuf(stdout,nullptr,_IOLBF,0);
 try{
  if(argc!=4)throw std::runtime_error("usage: driver baseline.elf candidate.elf outputdir");
  CK(hipInit(0));CK(hipSetDevice(0));int dev=-1;CK(hipGetDevice(&dev));if(dev!=0)throw std::runtime_error("GPU0 only");
  hipDeviceProp_t p{};CK(hipGetDeviceProperties(&p,0));if(std::strncmp(p.gcnArchName,"gfx906",6))throw std::runtime_error("gfx906 required");
  char pci[32]{};CK(hipDeviceGetPCIBusId(pci,sizeof(pci),0));if(std::strcmp(pci,"0000:05:00.0"))throw std::runtime_error("physical PCI mismatch");
  hipModule_t bm{},cm{};hipFunction_t b{},c[2]{};CK(hipModuleLoad(&bm,argv[1]));CK(hipModuleLoad(&cm,argv[2]));
  CK(hipModuleGetFunction(&b,bm,"_ZN6strata7prefill12_GLOBAL__N_120hc_plain_bf16_to_f16ILb0ELb1ELb1EEEvPKtP6__halflPjf"));
  CK(hipModuleGetFunction(&c[0],cm,"hc_exact_scale4096"));CK(hipModuleGetFunction(&c[1],cm,"hc_exact_scale512"));
  CK(hipMalloc(reinterpret_cast<void**>(&di),2*BATCH*(MAX_N+2)*2));CK(hipMalloc(reinterpret_cast<void**>(&doo),2*BATCH*(MAX_N+2)*2));CK(hipMalloc(reinterpret_cast<void**>(&ds),2*BATCH*ST*4));
  unsigned scales[]={4096,512},sizes[]={0,1,63,64,65,255,256,257,448};
  for(unsigned s=0;s<2;++s){for(unsigned first=0;first<65536;first+=BATCH){Batch v(0,scales[s],first,BATCH,1,0);run(v,b,c[s],argv[3]);}
   for(unsigned offset:{0u,5u})for(unsigned n:sizes){Batch v(1,scales[s],0,1,n,offset);run(v,b,c[s],argv[3]);}}
  CK(hipFree(ds));CK(hipFree(doo));CK(hipFree(di));CK(hipModuleUnload(cm));CK(hipModuleUnload(bm));
  if(launches!=TOTAL_LAUNCHES||batches!=548)throw std::runtime_error("coverage");
  std::puts("{\"type\":\"complete\",\"pairs\":131108,\"launches\":262216,\"batches\":548,\"passed\":true}");return 0;
 }catch(const std::exception&e){std::fputs("{\"type\":\"failure\",\"error\":",stdout);json_quote(e.what());std::printf(",\"launches\":%u,\"batches\":%u}\n",launches,batches);std::fprintf(stderr,"FAIL: %s; launches=%u\n",e.what(),launches);return 1;}
}
