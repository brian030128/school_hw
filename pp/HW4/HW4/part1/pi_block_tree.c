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
    long long int tosses = atoll(argv[1]);
    int world_rank, world_size;
    // ---

    // MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Each process calculates its own partial number of points in the circle
    long long int local_tosses = tosses / world_size;
    long long int local_number_in_circle = 0;
    
    // Seed the random number generator to be unique for each process
    srand(time(NULL) + world_rank);

    for (long long int toss = 0; toss < local_tosses; toss++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) {
            local_number_in_circle++;
        }
    }

    if (world_rank == 0) {
        long long int total_number_in_circle = local_number_in_circle;

        // --- TODO: Non-blocking receives ---
        // Allocate arrays for requests and the received data from worker processes.
        MPI_Request* requests = (MPI_Request*)malloc(sizeof(MPI_Request) * (world_size - 1));
        long long int* received_counts = (long long int*)malloc(sizeof(long long int) * (world_size - 1));

        // Issue a non-blocking receive for each worker process (ranks 1 to world_size-1).
        for (int i = 1; i < world_size; i++) {
            MPI_Irecv(
                &received_counts[i - 1], // Buffer for incoming data from rank i
                1,                       // Count of elements
                MPI_LONG_LONG,           // Datatype of the elements
                i,                       // Source rank
                0,                       // Message tag
                MPI_COMM_WORLD,          // Communicator
                &requests[i - 1]         // MPI_Request handle
            );
        }

        // Wait for all the non-blocking receive operations to complete.
        MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);

        // Aggregate the results from all workers.
        for (int i = 0; i < world_size - 1; i++) {
            total_number_in_circle += received_counts[i];
        }

        // Clean up allocated memory
        free(requests);
        free(received_counts);
        // ---

        // PI result
        pi_result = 4.0 * total_number_in_circle / ((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    } else {
        // --- TODO: All other ranks send their results to rank 0 ---
        MPI_Send(
            &local_number_in_circle, // Data to send
            1,                       // Count of elements
            MPI_LONG_LONG,           // Datatype
            0,                       // Destination rank (root)
            0,                       // Message tag
            MPI_COMM_WORLD           // Communicator
        );
        // ---
    }

    MPI_Finalize();
    return 0;
}