#pragma once
#include <cstdint>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <algorithm>
constexpr unsigned BATCH=256, ST=23, MAX_N=448, TOTAL_LAUNCHES=262216;
constexpr uint32_t GUARD=0xa55aa55a;
struct Args {
 uint16_t *in,*out; int64_t n; uint32_t*stats; float scale;
 void* pointers[5];
 Args(uint16_t*i,uint16_t*o,int64_t z,uint32_t*s,float f):in(i),out(o),n(z),stats(s),scale(f),pointers{&in,&out,&n,&stats,&scale}{}
 Args(const Args&)=delete;
};
struct Batch {
 uint32_t kind,scale,start,pairs,n,offset;
 std::vector<uint16_t> input,output;
 std::vector<uint32_t> stats;
 Batch(unsigned k,unsigned s,unsigned first,unsigned count,unsigned len,unsigned off):kind(k),scale(s),start(first),pairs(count),n(len),offset(off),input(count*2*(len+2)),output(input.size(),0xdead),stats(count*2*ST,GUARD){
  for(unsigned r=0;r<2*pairs;++r){
   unsigned at=r*(n+2);input[at]=output[at]=0xa55a;input[at+n+1]=output[at+n+1]=0x5aa5;
   for(unsigned j=0;j<n;++j){
    static const uint16_t edge[]={0,0x8000,0x0001,0x007f,0x0080,0x3f80,0xbf80,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc0,0x7f81,0xff81};
    input[at+1+j]=kind==0?uint16_t(start+r/2):(j<14?edge[j]:uint16_t(j*40503u+start));
   }
   std::fill(stats.begin()+r*ST+1+offset,stats.begin()+r*ST+17+offset,0);
  }
 }
 void save(const std::string&path)const{
  FILE*f=std::fopen(path.c_str(),"wb");if(!f)throw std::runtime_error("open raw binary");
  uint32_t h[]={0x48435031,kind,scale,start,pairs,n,offset};
  bool ok=std::fwrite(h,sizeof(h),1,f)==1 && std::fwrite(input.data(),2,input.size(),f)==input.size() && std::fwrite(output.data(),2,output.size(),f)==output.size() && std::fwrite(stats.data(),4,stats.size(),f)==stats.size();
  if(std::fclose(f)!=0||!ok)throw std::runtime_error("write raw binary");
 }
};
