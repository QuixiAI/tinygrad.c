#include <stdio.h>
#include "helpers/helpers.h"
int main(){
  const char* gzpath = "build/testdata.txt.gz";
  char out[512]={0};
  int rc = tg_fetch(gzpath, NULL, NULL, 1, 0, out, sizeof(out));
  printf("tg_fetch rc=%d out='%s'\n", rc, out);
  return 0;
}
