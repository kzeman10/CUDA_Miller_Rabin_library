CC = nvcc
CFLAGS = -std=c++20 -Xcompiler -fPIC

all: libmiller_rabin.so

libmiller_rabin.so: miller_rabin.o
	$(CC) $(CFLAGS) -shared -o libmiller_rabin.so miller_rabin.o

miller_rabin.o: miller_rabin.cu miller_rabin.h
	$(CC) $(CFLAGS) -c miller_rabin.cu

clean:
	rm -f libmiller_rabin.so miller_rabin.o
