#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm> // For std::min and std::copy

// TILE_SIZE is chosen to ensure the working set of three tiles (from A, B, and C)
// fits comfortably within the 384 KiB L1d cache.
// 3 * (128 * 128) * sizeof(int) = 192 KiB, which is well within the cache limit.
// This local optimization is highly effective for both small and large matrices.
constexpr int TILE_SIZE = 128;

// We can define a threshold to identify "large" matrices.
// While we use the same 1D algorithm, this allows us to be aware of the problem size.
constexpr int LARGE_MATRIX_THRESHOLD = 1000;

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // For large matrices, a 1D decomposition (replicating B) is less memory-scalable
    // than a 2D decomposition. However, it's simpler and still very effective for these problem sizes.
    if (rank == 0 && (n >= LARGE_MATRIX_THRESHOLD || m >= LARGE_MATRIX_THRESHOLD || l >= LARGE_MATRIX_THRESHOLD))
    {
        std::cout << "INFO: Using 1D block decomposition for large matrix set. Matrix B will be replicated." << std::endl;
    }

    // 1. Determine the number of rows of matrix A each process will handle (1D block decomposition).
    int rows_per_rank = n / size;
    int local_n = (rank == size - 1) ? (n - (size - 1) * rows_per_rank) : rows_per_rank;

    // 2. Allocate memory on each process.
    // Each process gets a horizontal strip of A and a full copy of B.
    // The input b_mat is already transposed, which is ideal for cache performance. We keep this layout.
    *a_mat_ptr = new int[local_n * m];
    *b_mat_ptr = new int[m * l];

    // 3. Distribute matrix A from rank 0 using MPI_Scatterv.
    // This is flexible for cases where n is not perfectly divisible by the number of processes.
    std::vector<int> sendcounts_a(size);
    std::vector<int> displs_a(size);

    if (rank == 0)
    {
        int current_displ = 0;
        for (int i = 0; i < size; ++i)
        {
            int rows_for_this_rank = (i == size - 1) ? (n - i * rows_per_rank) : rows_per_rank;
            sendcounts_a[i] = rows_for_this_rank * m;
            displs_a[i] = current_displ;
            current_displ += sendcounts_a[i];
        }
    }

    MPI_Scatterv(a_mat, sendcounts_a.data(), displs_a.data(), MPI_INT,
                 *a_mat_ptr, local_n * m, MPI_INT,
                 0, MPI_COMM_WORLD);

    // 4. Distribute matrix B.
    // On rank 0, copy b_mat to the student pointer. For other ranks, this buffer will be filled by the broadcast.
    if (rank == 0)
    {
        std::copy(b_mat, b_mat + (m * l), *b_mat_ptr);
    }
    // Broadcast the entire B matrix from rank 0 to all other processes.
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Determine local number of rows for this process.
    int rows_per_rank = n / size;
    int local_n = (rank == size - 1) ? (n - (size - 1) * rows_per_rank) : rows_per_rank;

    // 2. Allocate and initialize the local result matrix.
    std::vector<int> local_out_mat(local_n * l, 0);

    // 3. Perform tiled matrix multiplication on the local data.
    // The loop order is optimized for cache performance (i, j, k).
    // The tiling ensures that the actively used sub-matrices fit into the L1d cache.
    for (int i0 = 0; i0 < local_n; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < l; j0 += TILE_SIZE) {
            for (int k0 = 0; k0 < m; k0 += TILE_SIZE) {
                // Multiply the tiles: A[i0..i, k0..k] * B[k0..k, j0..j] -> C[i0..i, j0..j]
                for (int i = i0; i < std::min(i0 + TILE_SIZE, local_n); ++i) {
                    for (int j = j0; j < std::min(j0 + TILE_SIZE, l); ++j) {
                        int sum = local_out_mat[i * l + j];
                        for (int k = k0; k < std::min(k0 + TILE_SIZE, m); ++k) {
                            // Since b_mat is B-transposed, b_mat[j*m + k] accesses B[k][j].
                            // This results in sequential memory access for both a_mat and b_mat,
                            // which is ideal for performance.
                            sum += a_mat[i * m + k] * b_mat[j * m + k];
                        }
                        local_out_mat[i * l + j] = sum;
                    }
                }
            }
        }
    }

    // 4. Gather all local results into the final output matrix on rank 0 using MPI_Gatherv.
    std::vector<int> recvcounts_c(size);
    std::vector<int> displs_c(size);

    if (rank == 0)
    {
        int current_displ = 0;
        for (int i = 0; i < size; ++i)
        {
            int rows_for_this_rank = (i == size - 1) ? (n - i * rows_per_rank) : rows_per_rank;
            recvcounts_c[i] = rows_for_this_rank * l;
            displs_c[i] = current_displ;
            current_displ += recvcounts_c[i];
        }
    }

    MPI_Gatherv(local_out_mat.data(), local_n * l, MPI_INT,
                out_mat, recvcounts_c.data(), displs_c.data(), MPI_INT,
                0, MPI_COMM_WORLD);
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    // Free the memory allocated by each process in construct_matrices.
    delete[] a_mat;
    delete[] b_mat;
}