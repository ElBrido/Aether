CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math -funroll-loops -Iinclude
LDFLAGS = -lm

# Debug flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -Iinclude

# ASAN flags
ASAN_CFLAGS = -Wall -Wextra -g -O1 -fsanitize=address -fno-omit-frame-pointer -Iinclude
ASAN_LDFLAGS = -fsanitize=address -lm

SRC_DIR = src
OBJ_DIR = build
INC_DIR = include

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

TARGET = aether_engine

.PHONY: all clean debug asan

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

debug: clean
	$(MAKE) CFLAGS="$(DEBUG_CFLAGS)" LDFLAGS="$(LDFLAGS)"

asan: clean
	$(MAKE) CFLAGS="$(ASAN_CFLAGS)" LDFLAGS="$(ASAN_LDFLAGS)"

$(OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
