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

    // MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long int num_tosses_per_process = tosses / world_size;
    long long int local_number_in_circle = 0;

    // Seed the random number generator to get different results for each process
    unsigned int seed = world_rank * time(0);

    // Perform Monte Carlo simulation for the local number of tosses
    for (long long int toss = 0; toss < num_tosses_per_process; toss++)
    {
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1.0)
        {
            local_number_in_circle++;
        }
    }

    if (world_rank > 0)
    {
        // MPI workers: Send their local count to the master process (rank 0)
        MPI_Send(&local_number_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        long long int total_number_in_circle = local_number_in_circle;

        // non-blocking MPI communication.
        // If there is more than one process, receive results from workers.
        if (world_size > 1) {
            // Allocate arrays for requests and received data from worker nodes
            MPI_Request* requests = (MPI_Request*)malloc(sizeof(MPI_Request) * (world_size - 1));
            long long int* recv_counts = (long long int*)malloc(sizeof(long long int) * (world_size - 1));
            
            // Issue non-blocking receives for all worker processes
            for (int i = 1; i < world_size; i++)
            {
                MPI_Irecv(&recv_counts[i - 1], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
            }

            // Wait for all non-blocking receives to complete
            MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);

            // Aggregate the results from all workers
            for (int i = 0; i < world_size - 1; i++)
            {
                total_number_in_circle += recv_counts[i];
            }
            
            // Clean up allocated memory
            free(requests);
            free(recv_counts);
        }

        // PI result calculation
        pi_result = 4.0 * total_number_in_circle / ((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}