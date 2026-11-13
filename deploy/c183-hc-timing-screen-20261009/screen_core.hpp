#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <algorithm>
constexpr uint32_t GUARD=0xa55aa55a;constexpr unsigned ST=23,TOTAL=752,SCRUB_N=16777216;
struct Desc{unsigned n,scale,offset,calls;};
constexpr Desc DESCS[]={{3276800,4096,0,1728},{41943040,512,5,810},{41932800,512,5,54},{1310720,512,5,810},{1310400,512,5,54},{SCRUB_N,512,0,0}};
inline uint16_t pattern_at(unsigned pattern,unsigned i){
 constexpr uint16_t accepted[]={0x3e80,0xbe80,0x3f00,0xbf00,0x3f80,0xbf80,0x4000,0xc000,0x4080,0xc080,0x4100,0xc100};
 constexpr uint16_t reject[]={0x0001,0x7f7f,0x7f81,0x8000};
 return pattern==2?reject[i%4]:(pattern==1&&i%4096==0?reject[(i/4096)%4]:accepted[i%12]);
}
struct Fixture{
 unsigned descriptor,pattern;Desc d;std::vector<uint16_t> input;std::array<std::vector<uint16_t>,2> output;std::array<std::array<uint32_t,ST>,2> stats;
 Fixture(unsigned id,unsigned pat):descriptor(id),pattern(pat),d(DESCS[id]),input(d.n+2){
  input[0]=0xa55a;input[d.n+1]=0x5aa5;for(unsigned i=0;i<d.n;++i)input[i+1]=pattern_at(pat,i);
  for(unsigned arm=0;arm<2;++arm){output[arm].assign(d.n+2,0xdead);output[arm][0]=0xa55a;output[arm][d.n+1]=0x5aa5;reset_stats(arm);}
 }
 void reset_stats(unsigned arm){stats[arm].fill(GUARD);std::fill(stats[arm].begin()+1+d.offset,stats[arm].begin()+17+d.offset,0);}
 bool stats_ok(unsigned arm)const{for(unsigned j=0;j<ST;++j){uint32_t expected=(j<1+d.offset||j>=17+d.offset)?GUARD:(pattern&&j>=2+d.offset&&j<=4+d.offset?1:0);if(stats[arm][j]!=expected)return false;}return true;}
 bool check()const{
  if(input[0]!=0xa55a||input[d.n+1]!=0x5aa5)return false;
  for(unsigned i=0;i<d.n;++i)if(input[i+1]!=pattern_at(pattern,i))return false;
  for(unsigned arm=0;arm<2;++arm)if(output[arm][0]!=0xa55a||output[arm][d.n+1]!=0x5aa5||!stats_ok(arm))return false;
  return output[0]==output[1];
 }
 void save(const std::string&path,unsigned arm)const{
  FILE*f=std::fopen(path.c_str(),"wb");if(!f)throw std::runtime_error("open full evidence");uint32_t h[]={0x48535431,descriptor,pattern,arm,d.n,d.scale,d.offset};
  bool ok=std::fwrite(h,sizeof(h),1,f)==1&&std::fwrite(input.data(),2,input.size(),f)==input.size()&&std::fwrite(output[arm].data(),2,output[arm].size(),f)==output[arm].size()&&std::fwrite(stats[arm].data(),4,ST,f)==ST;
  if(std::fclose(f)!=0||!ok)throw std::runtime_error("write full evidence");
 }
};
struct Args{uint16_t*in,*out;int64_t n;uint32_t*stats;float scale;void*p[5];Args(uint16_t*i,uint16_t*o,int64_t count,uint32_t*s,float f):in(i),out(o),n(count),stats(s),scale(f),p{&in,&out,&n,&stats,&scale}{};Args(const Args&)=delete;};
