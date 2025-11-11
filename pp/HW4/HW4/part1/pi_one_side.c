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
    srand(world_rank * SEED);

    for (long long int i = 0; i < iterations_per_process; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            number_in_circle++;
        }
    }

    long long int total_number_in_circle = 0;
    MPI_Win win;

    // Create an MPI Window. This exposes the 'total_number_in_circle' variable on each process
    // for remote access. We will only be accessing the one on the root process (rank 0).
    MPI_Win_create(&total_number_in_circle, sizeof(long long int), sizeof(long long int), 
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Start a remote access epoch. This allows RMA operations to be issued.
    MPI_Win_fence(0, win);

    // Each process (including the root) accumulates its local 'number_in_circle'
    // into the 'total_number_in_circle' variable in the window of the root process (rank 0).
    // The displacement is 0 because the window points directly to the variable's address.
    MPI_Accumulate(&number_in_circle, 1, MPI_LONG_LONG, 0, 0, 1, MPI_LONG_LONG, MPI_SUM, win);

    // End the remote access epoch. This fence ensures that all MPI_Accumulate
    // operations have completed on the target (rank 0) before proceeding.
    MPI_Win_fence(0, win);

    if (world_rank == 0)
    {
        // Process PI result using the final aggregated count
        // which is now correctly summed in the local 'total_number_in_circle' variable.
        pi_result = 4.0 * total_number_in_circle / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    // Free the window object
    MPI_Win_free(&win);

    MPI_Finalize();
    return 0;
}