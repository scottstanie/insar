#include <endian.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEM_SIZE 3601 // Square DEM

int getIdx(int r, int c, int ncols) { return ncols * r + c; }
int16_t calcInterp(int16_t *demGrid, int i, int j, int bi, int bj, int rate);
int16_t interpRow(int16_t *demGrid, int i, int j, int bj, int rate);
int16_t interpCol(int16_t *demGrid, int i, int j, int bi, int rate);

int main(int argc, char **argv) {

  // Parse input filename, rate, and optional output filename
  if (argc < 3) {
    fprintf(stderr, "Usage: ./dem filename.hgt rate [outfilename.dem] \n");
    return EXIT_FAILURE;
  }
  char *filename = argv[1];
  int rate = atoi(argv[2]);

  // Optional input:
  const char *outfileUp;
  if (argc < 4) {
    outfileUp = "elevation_upsampled.dem";
    printf("Using %s as output file for upsampling.\n", outfileUp);
  } else {
    outfileUp = "elevation_upsampled.dem";
  }

  printf("Reading from %s\n", filename);
  printf("Upsampling by %d\n", rate);

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failure to open %s. Exiting.\n", filename);
    return EXIT_FAILURE;
  }

  int nbytes = 2;
  int16_t buf[1];
  int16_t *demGrid = (int16_t *)malloc(DEM_SIZE * DEM_SIZE * sizeof(*demGrid));
  printf("demGrid: %p\n", demGrid);

  int i = 0, j = 0;
  for (i = 0; i < DEM_SIZE; i++) {
    for (j = 0; j < DEM_SIZE; j++) {
      if (fread(buf, nbytes, 1, fp) != 1) {
        fprintf(stderr, "Read failure from %s\n", filename);
        return EXIT_FAILURE;
      }
      demGrid[getIdx(i, j, DEM_SIZE)] = be16toh(*buf);
    }
  }
  fclose(fp);

  // Interpolation
  short bi = 0, bj = 0;
  // Size of one side for upsampled
  // Example: 3 points at x = (0, 1, 2), rate = 2 becomes 5 points:
  //    x = (0, .5, 1, 1.5, 2)
  int upSize = rate * (DEM_SIZE - 1) + 1;
  printf("New size of upsampled DEM: %d\n", upSize);
  int16_t *upDemGrid = (int16_t *)malloc(upSize * upSize * sizeof(*upDemGrid));
  printf("upDemGrid: %p\n", demGrid);

  for (int i = 0; i < DEM_SIZE - 1; i++) {
    for (int j = 0; j < DEM_SIZE - 1; j++) {
      // At each point of the smaller DEM, walk bi, bj up to rate and find
      // interp value
      while (bi < rate) {
        int curBigi = rate * i + bi;
        while (bj < rate) {
          int16_t interpValue = calcInterp(demGrid, i, j, bi, bj, rate);
          int curBigj = rate * j + bj;
          upDemGrid[getIdx(curBigi, curBigj, upSize)] = interpValue;
          ++bj;
        }
        bj = 0; // reset the bj column back to 0 for this (i, j)
        ++bi;
      }
      bi = 0; // reset the bi row back to 0 for this (i, j)
    }
  }

  // Also must interpolate the last row/column: OOB for 2D interp, use 1D
  // Copy last col:
  bi = 0;
  for (i = 0; i < (DEM_SIZE - 1); i++) {
    j = (DEM_SIZE - 1); // Last col
    bj = 0;             // bj stays at 0 when j is max index
    int curBigj = rate * j + bj;
    while (bi < rate) {
      int16_t interpValue = interpCol(demGrid, i, j, bi, rate);
      int curBigi = rate * i + bi;
      upDemGrid[getIdx(curBigi, curBigj, upSize)] = interpValue;
      ++bi;
    }
    bi = 0; // reset the bi row back to 0 for this (i, j)
  }

  // Copy last row:
  bj = 0;
  for (j = 0; j < (DEM_SIZE - 1); j++) {
    i = (DEM_SIZE - 1); // Last row
    bi = 0;             // bi stays at 0 when i is max index
    int curBigi = rate * i + bi;
    while (bj < rate) {
      int16_t interpValue = interpRow(demGrid, i, j, bj, rate);
      int curBigj = rate * j + bj;
      upDemGrid[getIdx(curBigi, curBigj, upSize)] = interpValue;
      ++bj;
    }
    bj = 0; // reset the bj column back to 0 for this (i, j)
  }
  // Last, copy bottom right point
  upDemGrid[getIdx(upSize - 1, upSize - 1, upSize)] =
      demGrid[getIdx(DEM_SIZE - 1, DEM_SIZE - 1, DEM_SIZE)];

  printf("Finished with upsampling, writing to disk\n");

  fp = fopen(outfileUp, "wb");
  // fwrite(upDemGrid, sizeof(int16_t), upSize * upSize, fp);
  fwrite(upDemGrid, sizeof(int16_t), upSize * upSize, fp);
  fclose(fp);
  printf("%s write complete.\n", outfileUp);
  free(demGrid);
  free(upDemGrid);
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
  float x = (float)bj / rate;
  float y = (float)bi / rate;
  // Final result is cast back to int16_t by return type
  return a00 + (a10 * x) + (a01 * y) + (a11 * x * y);
}

int16_t interpRow(int16_t *demGrid, int i, int j, int bj, int rate) {
  // x is between 0 and 1: how far along row between orig points
  float x = (float)bj / rate;

  int16_t h1 = demGrid[getIdx(i, j, DEM_SIZE)];
  int16_t h2 = demGrid[getIdx(i, j + 1, DEM_SIZE)];

  return x * h2 + (1 - x) * h1;
}

int16_t interpCol(int16_t *demGrid, int i, int j, int bi, int rate) {
  // y is between 0 and 1: how far along column
  float y = (float)bi / rate;

  int16_t h1 = demGrid[getIdx(i, j, DEM_SIZE)];
  int16_t h2 = demGrid[getIdx(i + 1, j, DEM_SIZE)];

  return y * h2 + (1 - y) * h1;
}
