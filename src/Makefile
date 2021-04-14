
test: test.cu td_ins-td_del.cu sequential_heap.cpp
	nvcc test.cu -O3 td_ins-td_del.cu sequential_heap.cpp -std=c++17 -o test

clean:
	rm -rf a.out td_ins-td_del test libtd_ins-td_del.so