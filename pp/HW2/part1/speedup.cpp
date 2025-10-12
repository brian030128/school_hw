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

#define __AVX2_AVAILABLE__

pthread_mutex_t mutex_lock;

#define RADIUS (2<<14)

void *toss(void *arg) {
    long long thread_id = (long long)arg;
    long long tosses_per_thread = total_tosses / num_threads;
    if (thread_id == num_threads - 1)
        tosses_per_thread += total_tosses % num_threads;
    long long local_count = 0;

    SEFUtility::RNG::Xoshiro256PlusSIMD8 rng(123 + thread_id);
    alignas(32) float result[4];
    int runs = tosses_per_thread / 4 + (tosses_per_thread % 4 == 0 ? 0 : 1);

    const __m256 ones = _mm256_set1_ps(1.0f);
  for (long long i = 0; i < runs; i++) { // perform 8 toss at a time
    __m256 x = rng.next8().result_packed_;
    __m256 y = rng.next8().result_packed_;

    __m256 dist = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));
    __m256 in_circle_mask = _mm256_cmp_ps(dist, ones, _CMP_LE_OS);

    __m256 in_circle = _mm256_and_ps(ones, in_circle_mask); // a1, a2, a3, a4, a5, a6, a7, a8
    __m256 in_circle_permute = _mm256_permute2f128_ps(in_circle, in_circle, 1); // a5, a6, a7, a8, a1, a2, a3, a4

    in_circle = _mm256_hadd_ps(in_circle, in_circle_permute); // a1+a2, a3+a4, a5+a6, a7+a8, ....
    in_circle = _mm256_hadd_ps(in_circle, in_circle); // a1+a2+a3+a4, a5+a6+a7+a8, ....
    in_circle = _mm256_hadd_ps(in_circle, in_circle); // a1+a2+a3+a4+a5+a6+a7+a8, ....

    _mm256_store_ps(result, in_circle);
    // explicit conversion is important
    // long long 64-bit will be implicitly convert to float 32-bit if not specify
    local_count += (short) result[0];
  }

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
