#include "hc_counts_observer.hpp"
#include <cassert>
#include <memory>
#include <cstring>
using namespace hc_observer;
static const void*ptr(unsigned n){return reinterpret_cast<const void*>(uintptr_t(n)*4096+4096);}
static void bind(Registry&r,const void*o,unsigned t=4096,uint64_t p0=0,int dev=0,int lb=0,int le=27){r.chunk(o,t,p0,lb,le,dev,ptr(800),ptr(801),ptr(802));}
static void observation(Registry&r,Ticket t,unsigned w,unsigned x){unsigned g[10]{};for(unsigned j=0;j<3;++j){g[1+j]=(w>>j)&1;g[6+j]=(x>>j)&1;}r.eligible(t);r.observe(t,g,1000000);r.finish(t,((w|x)&3)?RangeReject:((w|x)&4)?ExactReject:Taken);}
int main(int argc,char**argv){
 assert(argc==2);auto reg=std::make_unique<Registry>();auto&r=*reg;
 if(std::strcmp(argv[1],"--fixture")==0){
  for(unsigned dev=0;dev<2;++dev){const void*o=ptr(1000+dev);r.start(o,-1,true,true);unsigned lo=dev?109:1,hi=dev?192:108;
   for(unsigned chunk=0;chunk<3;++chunk){unsigned t=chunk==2?4095:4096;bind(r,o,t,chunk*4096,dev,dev?27:0,dev?48:27);
    for(unsigned id=lo;id<=hi;++id){unsigned n=id%2?320:10240,k=id%2?10240:320;r.declare(o,ptr(id),id,n,k);auto ticket=r.attempt(o,ptr(id),n,k,t);const unsigned bad_ids[]={1,7,15,29,32,39,46,67,73,84,100,113,129,142,168,172,173,177,178,186};bool bad_weight=false;for(unsigned b:bad_ids)bad_weight|=id==b;observation(r,ticket,bad_weight?4:0,0);r.clear(o);}
   }auto*c=r.find(o);assert(c&&c->valid);r.flush(o,c->attempts,c->stages[Taken],stdout);
  }return 0;
 }
 const void*o=ptr(1000);r.start(o,-1,true,true);bind(r,o);r.declare(o,ptr(1),1,320,10240);
 auto t=r.attempt(o,ptr(1),320,10240,4096);observation(r,t,0,0);t=r.attempt(o,ptr(1),320,10240,4096);observation(r,t,4,0);
 assert(t.k->mask_changes==1&&t.k->bad_repeats_same_generation==0);
 t=r.attempt(o,ptr(1),320,10240,4096);observation(r,t,4,0);assert(t.k->bad_repeats_same_generation==1);
 r.rebind(o);t=r.attempt(o,ptr(1),320,10240,4096);observation(r,t,4,0);assert(t.k->repeats_changed_generation==1&&t.k->bad_repeats_same_generation==1);
 r.declare(o,ptr(2),1,320,10240);t=r.attempt(o,ptr(2),320,10240,4096);observation(r,t,4,0);assert(t.k->pointer_changes==1&&t.k->repeats_changed_generation==2);
 r.clear(o);auto bad=r.attempt(o,ptr(2),320,10240,4096);assert(!bad.k&&!bad.c->valid&&bad.c->unmapped==1);r.finish(bad,NoScratch);
 auto r2=std::make_unique<Registry>();r2->start(o,-1,true,true);bind(*r2,o);r2->declare(o,ptr(1),1,320,10240);t=r2->attempt(o,ptr(1),320,10240,4096);r2->rebind(o);r2->eligible(t);assert(!t.c->valid&&t.c->ticket_mismatches>=2);
 auto r3=std::make_unique<Registry>();r3->start(o,-1,true,true);bind(*r3,o);std::thread worker([&]{r3->declare(o,ptr(1),1,320,10240);});worker.join();assert(!r3->find(o)->valid&&r3->find(o)->thread_changes);
 auto r4=std::make_unique<Registry>();r4->start(o,-1,true,true);bind(*r4,o);r4->chunk(o,4096,4096,0,27,1,ptr(800),ptr(801),ptr(802));assert(!r4->find(o)->valid&&r4->find(o)->binding_changes==1);
 auto r5=std::make_unique<Registry>();FILE*f=std::tmpfile();assert(f);for(unsigned i=0;i<9;++i){r5->start(ptr(1000+i),-1,true,true);r5->flush(ptr(1000+i),0,0,f);}assert(r5->overflows==1&&r5->overflow_reported);std::fclose(f);
 auto r6=std::make_unique<Registry>();r6->start(o,-1,true,true);bind(*r6,o);for(unsigned i=0;i<257;++i)r6->declare(o,ptr(i),i+1,320,10240);assert(!r6->find(o)->valid&&r6->find(o)->overflows==1);
 auto r7=std::make_unique<Registry>();r7->start(o,-1,true,true);bind(*r7,o);r7->declare(o,ptr(1),1,320,10240);auto neg=r7->attempt(o,ptr(1),320,10240,4096);unsigned zero[10]{};r7->eligible(neg);r7->observe(neg,zero,-1);assert(!neg.c->valid&&neg.c->overflows&&neg.k->observation_count==1&&neg.k->observations[0].ns==-1);r7->finish(neg,Taken);
 assert(!r.find(o)->valid); // fourth/fifth timed observation exceeds the exact three-slot capacity
 std::puts("CPU observer lifecycle tests PASS");
}
