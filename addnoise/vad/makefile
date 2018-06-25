LIBS    := -lpcre2-8 -lrfe -liomp5 -lthrax -lfst -lfstfar -lr2ssp -lztvad

vadtest: vadtest.cpp
	g++ vadtest.cpp -std=c++0x -o vadtest -L./lib $(LIBS) -pthread -lrt
clean:
	rm -vf vadtest

