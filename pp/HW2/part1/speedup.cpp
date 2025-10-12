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

#define RADIUS (2<<14)

void *toss(void *number) {
  long long toss_num = *((long long *) number);
  long long in_circle_num = 0;

   SEFUtility::RNG::Xoshiro256Plus<SIMDInstructionSet::AVX2> rng(123 + thread_id);
  alignas(32) float result[8];

  for (long long i = 0; i < toss_num; i++) { // perform 8 toss at a time
    __m256 rand_float_x = _mm256_cvtepi32_ps(rng.next4().operator __m256i()); // convert random int to float
    __m256 float_x = _mm256_div_ps(rand_float_x, rand_max); // scale to -1 to 1

    __m256 rand_float_y = _mm256_cvtepi32_ps(rng.next4().operator __m256i()); // convert random int to float
    __m256 float_y = _mm256_div_ps(rand_float_y, rand_max); // scale to -1 to 1

    __m256 distance = _mm256_add_ps(_mm256_mul_ps(float_x, float_x), _mm256_mul_ps(float_y, float_y)); // x * x + y * y
    __m256 in_circle_mask = _mm256_cmp_ps(distance, one, _CMP_LE_OS); // distance <= 1

    __m256 in_circle = _mm256_and_ps(one, in_circle_mask); // a1, a2, a3, a4, a5, a6, a7, a8
    __m256 in_circle_permute = _mm256_permute2f128_ps(in_circle, in_circle, 1); // a5, a6, a7, a8, a1, a2, a3, a4

    in_circle = _mm256_hadd_ps(in_circle, in_circle_permute); // a1+a2, a3+a4, a5+a6, a7+a8, ....
    in_circle = _mm256_hadd_ps(in_circle, in_circle); // a1+a2+a3+a4, a5+a6+a7+a8, ....
    in_circle = _mm256_hadd_ps(in_circle, in_circle); // a1+a2+a3+a4+a5+a6+a7+a8, ....

    _mm256_store_ps(result, in_circle);
    // explicit conversion is important
    // long long 64-bit will be implicitly convert to float 32-bit if not specify
    in_circle_num += (short) result[0];
  }
      // Thread-safe update
    pthread_mutex_lock(&mutex_lock);
    global_count += in_circle_num;
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
