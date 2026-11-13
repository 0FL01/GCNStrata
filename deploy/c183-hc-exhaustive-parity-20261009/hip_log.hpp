#pragma once
#include <cstdio>
inline void json_quote(const char* s){
 std::putchar('"');
 for(const unsigned char*p=reinterpret_cast<const unsigned char*>(s);*p;++p){
  if(*p=='"'||*p=='\\'){std::putchar('\\');std::putchar(*p);}
  else if(*p<0x20)std::printf("\\u%04x",unsigned(*p));
  else std::putchar(*p);
 }
 std::putchar('"');
}
inline void emit_hip_rc(int rc,const char* op){
 std::fputs("{\"type\":\"hip_rc\",\"operation\":",stdout);json_quote(op);
 std::printf(",\"rc\":%d}\n",rc);std::fflush(stdout);
}

