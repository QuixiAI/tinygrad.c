#include "tqdm.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
  #include <windows.h>
  #include <io.h>
  #define ISATTY _isatty
  #define FD_STDERR 2
  static void _enable_vt(void){
    static int done=0; if(done) return; done=1;
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    if (h == INVALID_HANDLE_VALUE) return;
    DWORD mode = 0;
    if (GetConsoleMode(h, &mode)) {
      mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
      SetConsoleMode(h, mode);
    }
  }
#else
  #include <unistd.h>
  #define ISATTY isatty
  #define FD_STDERR 2
  static void _enable_vt(void){}
#endif

static struct {
  long total;
  long current;
  int  width;
  const char* prefix;
  int  enabled;
  long last_pct;
} _bar = {0, 0, 50, NULL, 0, -1};

static int _env_is_set(const char* k){ const char* v=getenv(k); return v && *v; }

void tg_tqdm_set_enabled(int enabled){ _bar.enabled = enabled; }
void tg_tqdm_set_width(int width){ _bar.width = (width>0 ? width : 50); }

void tg_tqdm_begin(const char* prefix, long total){
  _bar.total = (total>0 ? total : 0);
  _bar.current = 0;
  _bar.prefix = prefix ? prefix : "";
  _bar.width = (_bar.width>0 ? _bar.width : 50);
  _bar.last_pct = -1;
  _bar.enabled = ISATTY(FD_STDERR) && !_env_is_set("CI");
  _enable_vt();
  if (!_bar.enabled && _bar.prefix && *_bar.prefix){
    fprintf(stderr, "%s ...\n", _bar.prefix);
    fflush(stderr);
  }
}

void tg_tqdm_update(long current){
  if (!_bar.enabled) return;
  if (current < 0) current = 0;
  _bar.current = current;
  if (_bar.total > 0 && _bar.current > _bar.total) _bar.current = _bar.total;

  long pct = (_bar.total>0) ? (_bar.current * 100L) / _bar.total : -1;
  if (pct == _bar.last_pct && _bar.current != _bar.total) return;
  _bar.last_pct = pct;

  int width = _bar.width;
  int fill = (_bar.total>0) ? (int)((_bar.current * width) / _bar.total) : (int)(_bar.current % (width+1));

  fprintf(stderr, "\r\033[K");
  if (_bar.prefix && *_bar.prefix) fprintf(stderr, "%s ", _bar.prefix);
  fputc('[', stderr);
  for (int i=0;i<width;i++) fputc(i<fill ? '=' : ' ', stderr);
  fputc(']', stderr);
  if (_bar.total>0) fprintf(stderr, " %3ld%%", pct);
  fflush(stderr);

  if (_bar.total>0 && _bar.current == _bar.total){
    fputc('\n', stderr); fflush(stderr);
  }
}

void tg_tqdm_increment(long delta){ tg_tqdm_update(_bar.current + delta); }

void tg_tqdm_end(void){ if (_bar.enabled){ fprintf(stderr, "\n"); fflush(stderr);} }
