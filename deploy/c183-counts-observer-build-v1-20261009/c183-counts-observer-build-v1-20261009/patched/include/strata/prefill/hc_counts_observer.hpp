#pragma once
// Host-only bounded observer. No GPU API, device code, cache or route decisions.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
namespace hc_observer {
constexpr unsigned CONTEXTS=8,KEYS=256,STAGES=13;
enum Stage{NoScratch,SmallScratch,LdySmall,LdyLarge,YSizeOverflow,ScratchOverflow,XOverlap,WOverlap,YOverlap,RangeReject,ExactReject,ArenaOnly,Taken};
inline bool env_one(const char*n){const char*v=std::getenv(n);return v&&std::atoi(v)==1;}
inline bool enabled(){static const bool value=env_one("STRATA_HC_COUNTS_ONLY_OBSERVER");return value;}
inline bool contract(){return env_one("STRATA_GFX906_HC_F16")&&env_one("STRATA_HC_F16_SCALE_PROBE")&&env_one("STRATA_HC_F16_EXACT_INPUT")&&!env_one("STRATA_HC_F16_AUDIT")&&!env_one("STRATA_HC_SCALE_BOUNDS")&&!env_one("STRATA_HC_F16_SHADOW")&&!env_one("STRATA_HC_F16_ARENA_ONLY");}
struct Key {
 bool used=false,active=false,seen=false;const void*address=nullptr;unsigned ordinal=0,n=0,k=0,address_epoch=0,last_address_epoch=0,last_rebind=0,last_wmask=0;
 uint64_t declarations=0,pointer_changes=0,mask_changes=0,first_observations=0,repeats_same_generation=0,bad_repeats_same_generation=0,repeats_changed_generation=0;
 uint64_t attempts[2]{},eligible[2]{},stages[2][STAGES]{},masks[2][64]{};
};
struct Context {
 bool used=false,valid=true;const void*owner=nullptr;unsigned ordinal=0,generation=0,rebind_epoch=0;int device=-1;
 uint64_t attempts=0,eligible=0,guards=0,stages[STAGES]{},unmapped=0,ambiguous=0,overflows=0,device_query_errors=0;
 unsigned chunks=0;uint64_t chunk_T[8]{},chunk_p0[8]{};int layer_begin=-1,layer_end=-1;
 bool runtime_thread_set=false,bound=false;std::thread::id runtime_thread{};const void*model=nullptr,*session=nullptr,*stream=nullptr;
 uint64_t thread_changes=0,binding_changes=0,ticket_mismatches=0,inflight=0;
 std::array<Key,KEYS>keys{};
};
struct Ticket{Context*c=nullptr;Key*k=nullptr;unsigned t=0,generation=0,rebind=0,address_epoch=0;};
struct Registry {
 std::mutex mutex;std::array<Context,CONTEXTS>contexts{};unsigned generation=0;uint64_t overflows=0;bool overflow_reported=false;
 void claim(Context*c){auto id=std::this_thread::get_id();if(!c->runtime_thread_set){c->runtime_thread_set=true;c->runtime_thread=id;}else if(c->runtime_thread!=id){c->valid=false;++c->thread_changes;}}
 bool valid_ticket(Ticket t){claim(t.c);bool ok=t.c->used&&t.c->generation==t.generation&&t.c->rebind_epoch==t.rebind&&(!t.k||t.k->address_epoch==t.address_epoch);if(!ok){t.c->valid=false;++t.c->ticket_mismatches;}return ok;}
 Context*find(const void*owner){for(auto&c:contexts)if(c.used&&c.owner==owner)return &c;return nullptr;}
 void start(const void*owner,int device,bool query_ok,bool route_ok){std::lock_guard<std::mutex>lock(mutex);if(auto*c=find(owner)){c->valid=false;c->generation=++generation;return;}if(generation>=CONTEXTS){++overflows;return;}for(auto&c:contexts)if(!c.used){c=Context{};c.used=true;c.owner=owner;c.ordinal=unsigned(&c-contexts.data())+1;c.generation=++generation;c.device=device;c.valid=query_ok&&route_ok;c.device_query_errors=!query_ok;return;}++overflows;}
 void rebind(const void*owner){std::lock_guard<std::mutex>lock(mutex);if(auto*c=find(owner)){if(c->inflight){c->valid=false;++c->ticket_mismatches;}++c->rebind_epoch;}}
 void chunk(const void*owner,unsigned t,uint64_t p0,int lb,int le,int device,const void*model,const void*session,const void*stream){std::lock_guard<std::mutex>lock(mutex);if(auto*c=find(owner)){claim(c);if(c->bound&&(c->device!=device||c->model!=model||c->session!=session||c->stream!=stream)){c->valid=false;++c->binding_changes;}c->bound=true;c->device=device;c->model=model;c->session=session;c->stream=stream;if(c->chunks>=8){c->valid=false;++c->overflows;return;}if(c->chunks&& (c->layer_begin!=lb||c->layer_end!=le))c->valid=false;c->layer_begin=lb;c->layer_end=le;c->chunk_T[c->chunks]=t;c->chunk_p0[c->chunks]=p0;++c->chunks;}}
 void clear(const void*owner){std::lock_guard<std::mutex>lock(mutex);if(auto*c=find(owner))for(auto&k:c->keys)k.active=false;}
 void declare(const void*owner,const void*ptr,unsigned ordinal,unsigned n,unsigned k){
  std::lock_guard<std::mutex>lock(mutex);auto*c=find(owner);if(!c)return;claim(c);
  Key*slot=nullptr;for(auto&x:c->keys)if(x.used&&x.ordinal==ordinal){slot=&x;break;}
  if(!slot){for(auto&x:c->keys)if(!x.used){slot=&x;break;}}if(!slot){++c->overflows;c->valid=false;return;}
  if(!slot->used){slot->used=true;slot->address=ptr;slot->ordinal=ordinal;slot->n=n;slot->k=k;}
  else if(slot->n!=n||slot->k!=k){c->valid=false;}
  else if(slot->address!=ptr){slot->address=ptr;++slot->address_epoch;++slot->pointer_changes;}
  slot->active=true;++slot->declarations;
 }
 Ticket attempt(const void*owner,const void*ptr,unsigned n,unsigned k,unsigned t){
  std::lock_guard<std::mutex>lock(mutex);auto*c=find(owner);if(!c)return {};claim(c);if(!c->bound)c->valid=false;if(c->inflight)c->valid=false;++c->inflight;++c->attempts;Key*match=nullptr;unsigned matches=0;
  for(auto&x:c->keys)if(x.used&&x.active&&x.address==ptr&&x.n==n&&x.k==k){match=&x;++matches;}
  if(matches!=1){c->valid=false;if(matches==0)++c->unmapped;else ++c->ambiguous;match=nullptr;}
  if(t!=4095&&t!=4096){c->valid=false;return {c,nullptr,0,c->generation,c->rebind_epoch,0};}unsigned ti=t==4096;
  if(match){++match->attempts[ti];}
  return {c,match,ti,c->generation,c->rebind_epoch,match?match->address_epoch:0};
 }
 void eligible(Ticket t){if(!t.c)return;std::lock_guard<std::mutex>lock(mutex);if(!valid_ticket(t))return;++t.c->eligible;if(t.k)++t.k->eligible[t.t];}
 void finish(Ticket t,unsigned stage){if(!t.c)return;std::lock_guard<std::mutex>lock(mutex);if(!valid_ticket(t))return;if(t.c->inflight)--t.c->inflight;else t.c->valid=false;if(stage>=STAGES){t.c->valid=false;return;}++t.c->stages[stage];if(t.k)++t.k->stages[t.t][stage];}
 void observe(Ticket t,const unsigned*g){
  if(!t.c){return;}
  std::lock_guard<std::mutex>lock(mutex);if(!valid_ticket(t))return;++t.c->guards;unsigned w=0,x=0;
  for(unsigned j=0;j<3;++j){if(g[1+j]>1||g[6+j]>1)t.c->valid=false;w|=(g[1+j]!=0)<<j;x|=(g[6+j]!=0)<<j;}
  if(!t.k){return;}
  auto&k=*t.k;++k.masks[t.t][w*8+x];
  bool same=k.seen&&k.last_rebind==t.c->rebind_epoch&&k.last_address_epoch==k.address_epoch;
  if(!k.seen)++k.first_observations;else if(same){++k.repeats_same_generation;if(w&&k.last_wmask==w)++k.bad_repeats_same_generation;if(w!=k.last_wmask)++k.mask_changes;}else ++k.repeats_changed_generation;
  k.seen=true;k.last_rebind=t.c->rebind_epoch;k.last_address_epoch=k.address_epoch;k.last_wmask=w;
 }
 static void array(FILE*f,const uint64_t*v,unsigned n){std::fputc('[',f);for(unsigned j=0;j<n;++j)std::fprintf(f,"%s%llu",j?",":"",(unsigned long long)v[j]);std::fputc(']',f);}
 void flush(const void*owner,uint64_t original_attempts,uint64_t original_taken,FILE*f=stderr){
  std::lock_guard<std::mutex>lock(mutex);auto*c=find(owner);if(!c){if(overflows&&!overflow_reported){std::fputs("{\"type\":\"hc_observer_registry_overflow\"}\n",f);overflow_reported=true;}return;}
  uint64_t total=0,early=0,terminal=0;for(unsigned j=0;j<STAGES;++j){total+=c->stages[j];if(j<RangeReject)early+=c->stages[j];else terminal+=c->stages[j];}
  c->valid &= !overflows&&!c->inflight&&!c->thread_changes&&!c->binding_changes&&!c->ticket_mismatches&&c->attempts==original_attempts&&c->stages[Taken]==original_taken&&total==c->attempts&&early+c->eligible==c->attempts&&terminal==c->eligible&&c->guards==c->eligible;
  std::fprintf(f,"{\"type\":\"hc_observer_lifecycle\",\"context\":%u,\"generation\":%u,\"bound\":%s,\"thread_changes\":%llu,\"binding_changes\":%llu,\"ticket_mismatches\":%llu,\"inflight\":%llu}\n",c->ordinal,c->generation,c->bound?"true":"false",(unsigned long long)c->thread_changes,(unsigned long long)c->binding_changes,(unsigned long long)c->ticket_mismatches,(unsigned long long)c->inflight);
  std::fprintf(f,"{\"type\":\"hc_observer_summary\",\"context\":%u,\"generation\":%u,\"device\":%d,\"rebind_epoch\":%u,\"valid\":%s,\"attempts\":%llu,\"eligible\":%llu,\"guards\":%llu,\"original_attempts\":%llu,\"original_taken\":%llu,\"unmapped\":%llu,\"ambiguous\":%llu,\"overflows\":%llu,\"registry_overflows\":%llu,\"device_query_errors\":%llu,\"layer_begin\":%d,\"layer_end\":%d,\"chunk_T\":",c->ordinal,c->generation,c->device,c->rebind_epoch,c->valid?"true":"false",(unsigned long long)c->attempts,(unsigned long long)c->eligible,(unsigned long long)c->guards,(unsigned long long)original_attempts,(unsigned long long)original_taken,(unsigned long long)c->unmapped,(unsigned long long)c->ambiguous,(unsigned long long)c->overflows,(unsigned long long)overflows,(unsigned long long)c->device_query_errors,c->layer_begin,c->layer_end);array(f,c->chunk_T,c->chunks);std::fputs(",\"chunk_p0\":",f);array(f,c->chunk_p0,c->chunks);std::fputs(",\"stages\":",f);array(f,c->stages,STAGES);std::fputs("}\n",f);
  unsigned rows=0;for(const auto&k:c->keys)if(k.used){++rows;std::fprintf(f,"{\"type\":\"hc_observer_weight\",\"context\":%u,\"generation\":%u,\"tensor_ordinal\":%u,\"N\":%u,\"K\":%u,\"declarations\":%llu,\"address_epoch\":%u,\"pointer_changes\":%llu,\"within_generation_W_mask_changes\":%llu,\"first_observations\":%llu,\"repeats_same_visible_generation\":%llu,\"W_bad_repeats_same_visible_generation\":%llu,\"repeats_changed_visible_generation\":%llu,\"attempts_by_T\":",c->ordinal,c->generation,k.ordinal,k.n,k.k,(unsigned long long)k.declarations,k.address_epoch,(unsigned long long)k.pointer_changes,(unsigned long long)k.mask_changes,(unsigned long long)k.first_observations,(unsigned long long)k.repeats_same_generation,(unsigned long long)k.bad_repeats_same_generation,(unsigned long long)k.repeats_changed_generation);array(f,k.attempts,2);std::fputs(",\"eligible_by_T\":",f);array(f,k.eligible,2);std::fputs(",\"stages_4095\":",f);array(f,k.stages[0],STAGES);std::fputs(",\"stages_4096\":",f);array(f,k.stages[1],STAGES);std::fputs(",\"mask_hist_4095\":",f);array(f,k.masks[0],64);std::fputs(",\"mask_hist_4096\":",f);array(f,k.masks[1],64);std::fputs("}\n",f);}
  std::fprintf(f,"{\"type\":\"hc_observer_end\",\"context\":%u,\"generation\":%u,\"weight_rows\":%u}\n",c->ordinal,c->generation,rows);std::fflush(f);c->used=false;
 }
};
inline Registry&registry(){static Registry*r=new Registry;return *r;}
inline void start(const void*o){if(enabled())registry().start(o,-1,true,contract());}
inline void rebind(const void*o){if(enabled())registry().rebind(o);}
inline void chunk(const void*o,unsigned t,uint64_t p0,int lb,int le,int device,const void*model,const void*session,const void*stream){if(enabled())registry().chunk(o,t,p0,lb,le,device,model,session,stream);}
inline void clear_bindings(const void*o){if(enabled())registry().clear(o);}
inline void declare_weight(const void*o,const void*p,unsigned id,unsigned n,unsigned k){if(enabled())registry().declare(o,p,id,n,k);}
inline Ticket attempt(const void*o,const void*p,unsigned n,unsigned k,unsigned t){return enabled()?registry().attempt(o,p,n,k,t):Ticket{};}
inline void eligible(Ticket t){if(t.c)registry().eligible(t);}
inline void finish(Ticket t,Stage s){if(t.c)registry().finish(t,s);}
inline void observe(Ticket t,const unsigned*g){if(t.c)registry().observe(t,g);}
inline void flush(const void*o,uint64_t a,uint64_t taken){if(enabled())registry().flush(o,a,taken);}
}
