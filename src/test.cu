#include "td_ins-td_del.h"
#include "sequential_heap.h"
#include <bits/stdc++.h>
#include <ctime>

int arr[HEAP_CAPACITY];
int received_arr[HEAP_CAPACITY];
Heap *b = (Heap*)malloc(sizeof(Heap));
int number_of_streams = 20;
CPU_Heap my_heap(HEAP_CAPACITY);

void test_insertion()
{
    int n = HEAP_CAPACITY / BATCH_SIZE - 1;

    // create random elements for insertion
    for(int i = 0 ; i < HEAP_CAPACITY; i++)
        arr[i] = rand() % 10000000;
    
    std::clock_t c_start = clock(), c_end = clock();
    std::clock_t c_start_mem = clock(), c_end_mem = clock();

    // create multiple streams with non blocking flag
    cudaStream_t stream[number_of_streams];
    for(int i = 1 ; i < number_of_streams ; i++)
        cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
    
    // create copy of arr on device
    int *d_arr;
    gpuErrchk( cudaMalloc((void**)&d_arr, HEAP_CAPACITY * sizeof(int)));
    gpuErrchk( cudaMemcpy(d_arr, arr , HEAP_CAPACITY * sizeof(int), cudaMemcpyHostToDevice));
    c_end_mem = clock();
    long double time_elapsed_ms_mem = 1000.0 * (c_end_mem - c_start_mem) / CLOCKS_PER_SEC;

    c_start = std::clock();
    for(int i = 1; i <n  ; i++)
        td_insertion<<<1, BLOCK_SIZE,0, stream[i%(number_of_streams - 1) + 1]>>>(d_arr + i*BATCH_SIZE, BATCH_SIZE, d_heap_locks, d_partial_buffer, d_heap, i);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    c_end = std::clock();

    long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "GPU time for invocations: " << time_elapsed_ms << " ms\n";

    c_start_mem = clock();
    gpuErrchk( cudaMemcpy(b, d_heap, sizeof(Heap), cudaMemcpyDeviceToHost));
    c_end_mem = clock();
    time_elapsed_ms_mem += 1000.0 * (c_end_mem - c_start_mem) / CLOCKS_PER_SEC;
    std::cout << "GPU time for memcpy: " << time_elapsed_ms_mem << " ms\n";
    std::cout << "Total GPU time: " << time_elapsed_ms_mem + time_elapsed_ms << " ms\n";

    priority_queue<int, vector<int>, greater<int>> pq;
    c_start = std::clock();
    for(int i = 1; i < n ; i++)
        for(int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE ; j++)
            pq.push(arr[j]);
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU STL-Heap time: " << time_elapsed_ms << " ms\n";


    c_start = std::clock();
    for(int i = 1; i < n ; i++)
        for(int j = i * BATCH_SIZE; j < (i+1) * BATCH_SIZE ; j++)
            my_heap.push(arr[j]);
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU my-heap time used: " << time_elapsed_ms << " ms\n";

    // verify correctness
    bool correct = 1;
    for(int i = BATCH_SIZE ; i < 2*BATCH_SIZE ; i++)
    {
        int x = my_heap.pop();
        int y = pq.top();
        pq.pop();
        if (x != y){
            correct = 0;
        }
    }
    // correctness verified

    cout << ((correct)?"Success\n":"Failed!\n");
}

int main(int argc, char *argv[])
{   
    heap_init();
    test_insertion();
}