#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <random>
#include <atomic>

using namespace std;

long long total_tosses;


atomic<long long> global_count(0);
int num_threads;

#include "SIMDInstructionSet.h"

#include "Xoshiro256Plus.h"

#define __AVX2_AVAILABLE__

pthread_mutex_t mutex_lock;

#define RADIUS (2<<14)


void *toss(void *arg) {
    long long thread_id = (long long)arg;
    long long tosses_per_thread = total_tosses / num_threads;
    if (thread_id == num_threads - 1)
        tosses_per_thread += total_tosses % num_threads;
    double local_count = 0.0f;

    SEFUtility::RNG::Xoshiro256PlusSIMD8 rng(thread_id);
    alignas(32) float result[8];
    int runs = tosses_per_thread / 8 + (tosses_per_thread % 8 == 0 ? 0 : 1);

    const __m256 ones = _mm256_set1_ps(1.0f);
    for (long long i = 0; i < runs; i++) {
        __m256 x = rng.next8().result_packed_;
        __m256 y = rng.next8().result_packed_;

        __m256 dist = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));
        __m256 mask = _mm256_cmp_ps(dist, ones, _CMP_LE_OS);

        __m256 in_circle = _mm256_and_ps(ones, mask); 
        __m256 in_circle_permute = _mm256_permute2f128_ps(in_circle, in_circle, 1);

        in_circle = _mm256_hadd_ps(in_circle, in_circle_permute);
        in_circle = _mm256_hadd_ps(in_circle, in_circle);
        in_circle = _mm256_hadd_ps(in_circle, in_circle);

        _mm256_store_ps(result, in_circle);
        local_count += result[0];
    }

    global_count.fetch_add((long long)local_count, memory_order_relaxed);
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
    cout << pi_estimate << endl;

    return 0;
}
