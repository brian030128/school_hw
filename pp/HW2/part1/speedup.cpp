#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <random>   // for std::mt19937_64
#include <chrono>

using namespace std;

long long total_tosses;
long long global_count = 0;
int num_threads;

pthread_mutex_t mutex_lock;

void* toss(void* arg) {
    long long thread_id = (long long)arg;
    long long tosses_per_thread = total_tosses / num_threads;
    if (thread_id == num_threads - 1) {
        // The last thread takes any remainder
        tosses_per_thread += total_tosses % num_threads;
    }

    // Use a high-quality 64-bit random generator
    std::mt19937_64 gen(chrono::steady_clock::now().time_since_epoch().count() + thread_id);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    long long local_count = 0;
    for (long long i = 0; i < tosses_per_thread; i++) {
        double x = dist(gen);
        double y = dist(gen);
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
        cerr << "Error: Both arguments must be positive numbers." << endl;
        return 1;
    }

    pthread_t* threads = new pthread_t[num_threads];
    pthread_mutex_init(&mutex_lock, nullptr);

    // Create threads
    for (long long i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], nullptr, toss, (void*)i);
    }

    // Wait for threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    pthread_mutex_destroy(&mutex_lock);

    double pi_estimate = 4.0 * (double)global_count / (double)total_tosses;
    cout.precision(12);
    cout << "Estimated Pi = " << pi_estimate << endl;

    delete[] threads;
    return 0;
}
