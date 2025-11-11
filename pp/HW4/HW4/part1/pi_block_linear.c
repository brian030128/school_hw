#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>


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
    if (world_rank > 0)
    {
        // Workers send their intermediate count to the master process (rank 0)
        MPI_Send(&number_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // Master process receives intermediate counts from all workers
        long long int total_number_in_circle = number_in_circle;
        long long int received_count;
        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&received_count, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_number_in_circle += received_count;
        }
        number_in_circle = total_number_in_circle;
    }

    if (world_rank == 0)
    {
        // Process PI result
        pi_result = 4.0 * number_in_circle / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
