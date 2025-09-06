// tqdm.h - minimal progress bar (stderr) with TTY/CI gating
#ifndef TQDM_H
#define TQDM_H

#ifdef __cplusplus
extern "C" {
#endif

void tg_tqdm_begin(const char* prefix, long total);
void tg_tqdm_update(long current);
void tg_tqdm_increment(long delta);
void tg_tqdm_end(void);
void tg_tqdm_set_width(int width);
void tg_tqdm_set_enabled(int enabled);

#ifdef __cplusplus
}
#endif

#endif // TQDM_H

