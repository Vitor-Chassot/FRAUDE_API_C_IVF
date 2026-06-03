CC = gcc

CFLAGS = \
-O3 \
-march=haswell \
-mtune=haswell \
-mavx2 \
-mfma \
-flto \
-ffast-math \
-fomit-frame-pointer \
-falign-functions=32 \
-falign-loops=32 \
-funroll-loops \
-fno-plt \
-pthread \
-D_GNU_SOURCE \
-DNDEBUG \
-std=gnu11 \
-Wall \
-Wextra \
-Wshadow 

LDFLAGS = \
-flto \
-pthread \
-lm 

SRC = \
src/api.c \
src/payload_vectorizer.c \
src/knn_classifier.c

TARGET = fraude-api

all:
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)
	strip -s $(TARGET)

clean:
	rm -f $(TARGET)
