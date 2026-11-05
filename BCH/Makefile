CXX ?= g++
AR ?= ar
CPPFLAGS += -I.
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow

.PHONY: all test example sanitize clean
all: build/libbch512.a build/recover_example

build:
	mkdir -p $@

build/BCH.o: BCH.cpp BCH.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c BCH.cpp -o $@

build/libbch512.a: build/BCH.o
	$(AR) rcs $@ $<

build/recover_example: examples/recover_example.cpp BCH.hpp build/libbch512.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< build/libbch512.a -o $@

build/test_bch: tests/test_bch.cpp BCH.hpp build/libbch512.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< build/libbch512.a -o $@

test: build/test_bch
	./build/test_bch

example: build/recover_example
	./build/recover_example

build/test_bch_sanitize: tests/test_bch.cpp BCH.cpp BCH.hpp | build
	$(CXX) $(CPPFLAGS) -std=c++17 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer tests/test_bch.cpp BCH.cpp -o $@

sanitize: build/test_bch_sanitize
	./build/test_bch_sanitize

clean:
	rm -rf build
