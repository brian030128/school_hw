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

pthread_mutex_t mutex_lock;

thread_local std::mt19937 gen(std::random_device{}());
thread_local std::uniform_int_distribution<int> dist(1, 1000);

#define RADIUS_SQUARE 1000000 

void* toss(void* arg) {
    long long thread_id = (long long)arg;
    long long tosses_per_thread = total_tosses / num_threads;
    if (thread_id == num_threads - 1)
        tosses_per_thread += total_tosses % num_threads;

    long long local_count = 0;
    for (long long i = 0; i < tosses_per_thread; i++) {
        int x = dist(gen);
        int y = dist(gen);
        if (x * x + y * y <= RADIUS_SQUARE)
            local_count++;
    }

    global_count += local_count;

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
