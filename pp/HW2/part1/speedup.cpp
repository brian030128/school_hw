#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>

using namespace std;

long long total_tosses;
long long global_count = 0;
int num_threads;

pthread_mutex_t mutex_lock;

struct XorShift128Plus {
    uint64_t s[2];
    inline uint64_t next() {
        uint64_t x = s[0];
        const uint64_t y = s[1];
        s[0] = y;
        x ^= x << 23;
        s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s[1] + y;
    }
    inline double next_double() {
        // Convert to [0,1)
        return (next() >> 11) * (1.0 / 9007199254740992.0); // 2^53
    }
};

void* toss(void* arg) {
    long long thread_id = (long long)arg;
    long long tosses_per_thread = total_tosses / num_threads;
    if (thread_id == num_threads - 1)
        tosses_per_thread += total_tosses % num_threads;

    XorShift128Plus rng;
    uint64_t seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.s[0] = seed ^ (thread_id + 0x9e3779b97f4a7c15ULL);
    rng.s[1] = seed ^ (thread_id * 0xbf58476d1ce4e5b9ULL);

    long long local_count = 0;
    for (long long i = 0; i < tosses_per_thread; i++) {
        double x = rng.next_double() * 2.0 - 1.0;
        double y = rng.next_double() * 2.0 - 1.0;
        if (x * x + y * y <= 1.0)
            local_count++;
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
