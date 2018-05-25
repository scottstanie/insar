#include <endian.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DEM_SIZE 3601 // Square DEM

int getIdx(int r, int c, int ncols) { return ncols * r + c; }
int16_t calcInterp(int16_t *demGrid, int i, int j, int bi, int bj, int rate);

int main(int argc, char **argv) {

  // Parse input filename and rate
  if (argc < 3) {
    fprintf(stderr, "Usage: ./dem filename.hgt rate \n");
    return EXIT_FAILURE;
  }
  char *filename = argv[1];
  int rate = atoi(argv[2]);

  printf("Reading from %s\n", filename);
  printf("Upsampling by %d\n", rate);

  FILE *fp = fopen(filename, "r");
  if (fp == NULL)
    return EXIT_FAILURE;

  int nbytes = 2;
  int16_t buf[1];
  int16_t *demGrid = (int16_t *)malloc(DEM_SIZE * DEM_SIZE * sizeof(*demGrid));

  for (int i = 0; i < DEM_SIZE; i++) {
    for (int j = 0; j < DEM_SIZE; j++) {
      if (fread(buf, nbytes, 1, fp) != 1) {
        fprintf(stderr, "Read failure from %s\n", filename);
        printf("Read failure from %s\n", filename);
        return EXIT_FAILURE;
      }
      demGrid[getIdx(i, j, DEM_SIZE)] = be16toh(*buf);
    }
  }
  fclose(fp);

  fp = fopen("out.dem", "wb");
  fwrite(demGrid, sizeof(int16_t), DEM_SIZE * DEM_SIZE, fp);
  fclose(fp);

  // Interpolation
  short bi = 0, bj = 0;
  // Size of one side for upsampled
  // Example: 3 points at x = (0, 1, 2), rate = 2 becomes 5 points:
  //    x = (0, .5, 1, 1.5, 2)
  int upSize = rate * (DEM_SIZE - 1) + 1;
  printf("New size of upsampled DEM: %d\n", upSize);
  int16_t *upDemGrid = (int16_t *)malloc(upSize * upSize * sizeof(*upDemGrid));

  for (int i = 0; i < DEM_SIZE - 1; i++) {
    for (int j = 0; j < DEM_SIZE - 1; j++) {
      while (bi < rate) {
        short curBigi = rate * i + bi;
        while (bj < rate) {
          int16_t interpValue = calcInterp(demGrid, i, j, bi, bj, rate);
          short curBigj = rate * j + bj;
          upDemGrid[getIdx(curBigi, curBigj, upSize)] = interpValue;
          ++bj;
        }
        bj = 0; // reset the bj column back to 0 for this (i, j)
        ++bi;
      }
      bi = 0; // reset the bi row back to 0 for this (i, j)
    }
  }

  // Finally, copy over the last row and last column
  // Copy last row:
  for (int i = 0; i < DEM_SIZE; i++) {
    int j = (DEM_SIZE - 1); // Last row
    while (bi < rate) {
      short curBigi = rate * i + bi;
      while (bj < rate) {
        short curBigj = rate * j + bj;
        upDemGrid[getIdx(curBigi, curBigj, upSize)] =
            demGrid[getIdx(i, j, DEM_SIZE)];
        ++bj;
      }
      bj = 0; // reset the bj column back to 0 for this (i, j)
      ++bi;
    }
    bi = 0; // reset the bi row back to 0 for this (i, j)
  }
  // Copy last column:
  for (int j = 0; j < DEM_SIZE; j++) {
    int i = (DEM_SIZE - 1); // Last col
    while (bi < rate) {
      short curBigi = rate * i + bi;
      while (bj < rate) {
        short curBigj = rate * j + bj;
        upDemGrid[getIdx(curBigi, curBigj, upSize)] =
            demGrid[getIdx(i, j, DEM_SIZE)];
        ++bj;
      }
      bj = 0; // reset the bj column back to 0 for this (i, j)
      ++bi;
    }
    bi = 0; // reset the bi row back to 0 for this (i, j)
  }

  fp = fopen("out_up.dem", "wb");
  fwrite(upDemGrid, sizeof(int16_t), upSize * upSize, fp);
  fclose(fp);
  return 0;
}

int16_t calcInterp(int16_t *demGrid, int i, int j, int bi, int bj, int rate) {
  int16_t h1 = demGrid[getIdx(i, j, DEM_SIZE)];
  int16_t h2 = demGrid[getIdx(i, j + 1, DEM_SIZE)];
  int16_t h3 = demGrid[getIdx(i + 1, j, DEM_SIZE)];
  int16_t h4 = demGrid[getIdx(i + 1, j + 1, DEM_SIZE)];

  int a00 = h1;
  int a10 = h2 - h1;
  int a01 = h3 - h1;
  int a11 = h1 - h2 - h3 + h4;
  // x and y are between 0 and 1: how far in the 1x1 cell we are
  float x = bj / rate;
  float y = bi / rate;
  return a00 + (a10 * x) + (a01 * y) + (a11 * x * y);
}
