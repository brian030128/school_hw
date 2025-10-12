#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <random>

using namespace std;

long long total_tosses;
long long global_count = 0;
int num_threads;

#include "SIMDInstructionSet.h"

#include "Xoshiro256Plus.h"

pthread_mutex_t mutex_lock;

#define RADIUS (2<<15)

void* toss(void* arg) {
    long long thread_id = (long long)arg;
    long long tosses_per_thread = total_tosses / num_threads;
    if (thread_id == num_threads - 1)
        tosses_per_thread += total_tosses % num_threads;

    long long local_count = 0;
    SEFUtility::RNG::Xoshiro256Plus<SIMDInstructionSet::AVX2> rng(123 + thread_id);

    // Process 4 tosses at a time
    long long i;
    for (i = 0; i + 3 < tosses_per_thread; i += 4) {
        auto rng_values = rng.next4(0, RADIUS);
        
        // Process all 4 values
        for (int j = 0; j < 4; j++) {
            int x = rng_values[j * 2];      // x values at even indices
            int y = rng_values[j * 2 + 1];  // y values at odd indices
            if (x * x + y * y <= RADIUS * RADIUS)
                local_count++;
        }
    }
    
    // Handle remaining tosses (if tosses_per_thread not divisible by 4)
    for (; i < tosses_per_thread; i++) {
        int x = rng.next(0, RADIUS);
        int y = rng.next(0, RADIUS);
        if (x * x + y * y <= RADIUS * RADIUS)
            local_count++;
    }

    // Thread-safe update
    pthread_mutex_lock(&mutex_lock);
    global_count += local_count;
    pthread_mutex_unlock(&mutex_lock);

    return nullptr;
}
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <number_of_threads> <number_of_tosses>" << endl;
        return 1;
    }

    num_threads = atoi(argv[1]);
    total_tosses = atoll(argv[2]);
    if (num_threads <= 0 || total_tosses <= 0) {
        cerr << "Error: Both arguments must be positive." << endl;
        return 1;
    }

    pthread_t* threads = new pthread_t[num_threads];
    pthread_mutex_init(&mutex_lock, nullptr);

    auto start = chrono::high_resolution_clock::now();

    for (long long i = 0; i < num_threads; i++)
        pthread_create(&threads[i], nullptr, toss, (void*)i);

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], nullptr);

    auto end = chrono::high_resolution_clock::now();
    double seconds = chrono::duration<double>(end - start).count();

    pthread_mutex_destroy(&mutex_lock);
    delete[] threads;

    double pi_estimate = 4.0 * (double)global_count / (double)total_tosses;
    cout.precision(12);
    cout << "Estimated Pi = " << pi_estimate << endl;
    cout << "Execution Time: " << seconds << " sec" << endl;

    return 0;
}
