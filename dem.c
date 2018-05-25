#include <endian.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DEM_SIZE 3601 // Square DEM

int getIdx(int r, int c, int ncols) { return ncols * r + c; }

int main(int argc, char **argv) {
  int nbytes = 2;
  int16_t buf[1];
  printf("%lu\n", sizeof(int16_t));

  const char *filename = (argc > 1) ? argv[1] : "t.dem";
  printf("Reading from %s\n", filename);
  FILE *fp = fopen(filename, "r");
  if (fp == NULL)
    return EXIT_FAILURE;

  int16_t *demArray = (int16_t *)malloc(DEM_SIZE * DEM_SIZE * sizeof(int16_t));
  for (int i = 0; i < DEM_SIZE; i++) {
    for (int j = 0; j < DEM_SIZE; j++) {
      if (fread(buf, nbytes, 1, fp) != 1) {
        return EXIT_FAILURE;
      }
      demArray[getIdx(i, j, DEM_SIZE)] = be16toh(*buf);
    }
  }
  fclose(fp);

  FILE *fpOut = fopen("out.dem", "wb");
  // for (int i = 0; i < DEM_SIZE * DEM_SIZE; i++)
  fwrite(demArray, sizeof(int16_t), DEM_SIZE * DEM_SIZE, fpOut);
  fclose(fpOut);

  // while (fread(buf, nbytes, 1, fp) > 0) {
  // printf("%d\n", (int16_t)*buf);
  // num = be16toh(*buf);
  // printf("%d\n", num);
  // printf("-----\n");
  // }

  return 0;
}
