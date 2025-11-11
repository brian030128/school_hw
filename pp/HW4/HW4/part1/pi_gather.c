#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define SEED 12345678

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // init MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long int number_in_circle = 0;
    long long int iterations_per_process = tosses / world_size;
    unsigned int seed = world_rank * time(0);

    for (long long int i = 0; i < iterations_per_process; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            number_in_circle++;
        }
    }
    
    // Create a buffer on the root process to hold the gathered counts
    long long int *gather_buffer = NULL;
    if (world_rank == 0) {
        gather_buffer = malloc(world_size * sizeof(long long int));
    }

    // Gather all the 'number_in_circle' counts from each process to the root process
    MPI_Gather(&number_in_circle, 1, MPI_LONG_LONG, gather_buffer, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // Aggregate the intermediate counts from the gather buffer
        long long int total_number_in_circle = 0;
        for (int i = 0; i < world_size; i++) {
            total_number_in_circle += gather_buffer[i];
        }

        // Free the buffer
        free(gather_buffer);

        // Process PI result
        pi_result = 4.0 * total_number_in_circle / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}