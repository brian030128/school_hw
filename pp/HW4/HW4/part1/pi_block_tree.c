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
    long long int tosses = atoll(argv[1]); // Use atoll for long long int
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Each process calculates its own partial number of points in the circle.
    long long int tosses_per_process = tosses / world_size;
    long long int number_in_circle = 0;
    
    // Seed the random number generator uniquely for each process.
    srand(time(NULL) + world_rank);

    for (long long int i = 0; i < tosses_per_process; i++)
    {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1)
        {
            number_in_circle++;
        }
    }

    // TODO: binary tree redunction
    // The reduction happens in log2(world_size) steps.
    // In each step, the distance between communicating processes doubles.
    for (int distance = 1; distance < world_size; distance *= 2)
    {
        // A process is a "sender" if its rank is in the upper half of a sub-group.
        // The condition (world_rank % (2 * distance)) == distance identifies these processes.
        if ((world_rank % (2 * distance)) == distance)
        {
            int destination = world_rank - distance;
            MPI_Send(&number_in_circle, 1, MPI_LONG_LONG, destination, 0, MPI_COMM_WORLD);
            // Once a process has sent its data, it becomes inactive for the rest of the reduction.
            break;
        }
        // A process is a "receiver" if its rank is the base of a sub-group.
        // The condition (world_rank % (2 * distance)) == 0 identifies these processes.
        else if ((world_rank % (2 * distance)) == 0)
        {
            // The receiver must ensure its sending partner exists.
            // This is always true in a power-of-two scenario but is good practice.
            int source = world_rank + distance;
            if (source < world_size)
            {
                long long int received_count;
                MPI_Recv(&received_count, 1, MPI_LONG_LONG, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                number_in_circle += received_count;
            }
        }
    }


    if (world_rank == 0)
    {
        // TODO: PI result
        // After the reduction, rank 0 has the total count from all processes.
        pi_result = 4.0 * number_in_circle / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}