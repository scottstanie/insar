int upsample(const char *filename, const int rate, const long ncols,
             const long nrows, const char *outfileUp);
#include <stdint.h>

int getIdx(int r, int c, int ncols) { return ncols * r + c; }
const char *getFileExt(const char *filename);
int16_t calcInterp(int16_t *demGrid, int i, int j, int bi, int bj, int rate,
                   int ncols);
int16_t interpRow(int16_t *demGrid, int i, int j, int bj, int rate, int ncols);
int16_t interpCol(int16_t *demGrid, int i, int j, int bi, int rate, int ncols);
