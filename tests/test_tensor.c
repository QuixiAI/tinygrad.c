#include "tg.h"
#include <assert.h>
#include <stdio.h>
int main(){
  tg_ctx_t ctx; assert(tgCreateContext(&ctx) == 0);
  int64_t shape[2] = {2,3};
  tg_tensor_t t; assert(tgTensorCreate(ctx, TG_F32, shape, 2, &t) == 0);
  float data[6] = {1,2,3,4,5,6};
  assert(tgTensorUpload(t, data, sizeof(data)) == 0);
  float out[6] = {0};
  assert(tgTensorDownload(t, out, sizeof(out)) == 0);
  for (int i=0;i<6;i++) assert(out[i]==data[i]);
  assert(tgTensorDestroy(t) == 0);
  assert(tgDestroyContext(ctx) == 0);
  printf("test_tensor: OK\\n");
  return 0;
}
