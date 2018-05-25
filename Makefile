BUILD_DIR = bin
SRC_DIR = insar
CC = gcc
CFLAGS = -g -Wall -std=gnu99 -O3
MKDIR_P = mkdir -p

TARGET = $(BUILD_DIR)/upsample

SRCS = $(wildcard $(SRC_DIR)/*.c)
# OBJS = $(patsubst %.c, $(BUILD_DIR)/%.o, $(wildcard $(SRC_DIR)/*.c))

default: $(TARGET)
all: default


$(TARGET): $(SRCS)
	$(CC) $(SRCS) $(CFLAGS) -o $@


.PHONY: clean

clean:
	rm -f *.o
	rm -f $(TARGET)


