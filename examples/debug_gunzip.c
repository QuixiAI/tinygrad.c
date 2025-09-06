#include <stdio.h>
#include "compat/fetch.h"
int main(){
  const char* in = "build/testdata.txt.gz";
  const char* out = "build/testdata.txt.gz.gunzip_dbg";
  int rc = tg_gunzip_impl(in, out);
  printf("tg_gunzip_impl rc=%d\n", rc);
  return rc!=0;
}
