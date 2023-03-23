#
# Makefile for lab3 in Advanced Computer Architecture.
#
# Author: Andreas Sandberg <andreas.sandberg@it.uu.se>
#
#
CC=gcc
CFLAGS=-std=c11 -O3 -D_XOPEN_SOURCE=600 -Wall -Werror -g
LDFLAGS=
LIBS=-lm -lrt -latomic

all: gs_seq gs_pth

gs_pth: gs_common.o gsi_pth.o timing.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS) -lpthread

gs_seq: gs_common.o gsi_seq.o timing.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

test: gs_seq gs_pth
	$(eval TMP_SEQ := $(shell mktemp --suffix=.gs_seq.test.out))
	$(eval TMP_PTH := $(shell mktemp --suffix=.gs_pth.test.out))

	@echo '**********************************************************************'
	@echo 'Starting sequential reference run...'
	@echo '**********************************************************************'
	./gs_seq -s 512 -o $(TMP_SEQ)
	@echo

	@echo '**********************************************************************'
	@echo 'Starting parallel run...'
	@echo '**********************************************************************'
	./gs_pth -s 512 -o $(TMP_PTH)
	@echo

	@echo "Test results: "
	@if diff -q "$(TMP_SEQ)" "$(TMP_PTH)" >/dev/null; then \
		echo 'OK'; \
		rm -f $(TMP_SEQ) $(TMP_PTH); \
	else \
		echo 'MISMATCH'; \
		rm -f $(TMP_SEQ) $(TMP_PTH); \
		exit 1; \
	fi


clean:
	rm -f gs_seq gs_pth *.o

.PHONY: all test clean
