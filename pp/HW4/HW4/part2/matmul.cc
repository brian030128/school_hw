#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm> // For std::min and std::copy

// TILE_SIZE is chosen to ensure the working set of three tiles (from A, B, and C)
// fits comfortably within the L1d cache.
// 3 * (128 * 128) * sizeof(int) = 192 KiB.
constexpr int TILE_SIZE = 128;

// --- construct_matrices and destruct_matrices remain unchanged ---

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(n < 1000){
        construct_matrices_swizzle(n,m,l,a_mat, b_mat,a_mat_ptr, b_mat_ptr);
    }

    int rows_per_rank = n / size;
    int local_n = (rank == size - 1) ? (n - (size - 1) * rows_per_rank) : rows_per_rank;

    *a_mat_ptr = new int[local_n * m];
    *b_mat_ptr = new int[m * l];

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

    if (rank == 0)
    {
        std::copy(b_mat, b_mat + (m * l), *b_mat_ptr);
    }
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size < 1000){
        matrix_multiply_swizzle(n,m,l,a_mat, b_mat, out_mat);
        return;
    }

    int rows_per_rank = n / size;
    int local_n = (rank == size - 1) ? (n - (size - 1) * rows_per_rank) : rows_per_rank;

    std::vector<int> local_out_mat(local_n * l, 0);

    // Perform tiled matrix multiplication using the standard, high-performance i0-j0-k0 loop order.
    // This order maximizes cache reuse for the result matrix (local_out_mat), which is critical
    // because it involves both reads and writes.
    for (int i0 = 0; i0 < local_n; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < l; j0 += TILE_SIZE) {
            // The two outer loops select a single tile from the C matrix (local_out_mat).
            // This tile will be held in cache while it is being computed.
            for (int k0 = 0; k0 < m; k0 += TILE_SIZE) {
                // This innermost tiling loop iterates through the necessary A and B tiles
                // to fully compute the selected C tile.
                for (int i = i0; i < std::min(i0 + TILE_SIZE, local_n); ++i) {
                    for (int j = j0; j < std::min(j0 + TILE_SIZE, l); ++j) {
                        // The `sum` variable is crucial. It accumulates the dot-product result
                        // in a CPU register, avoiding repeated memory access to local_out_mat[i*l+j].
                        int sum = local_out_mat[i * l + j];
                        for (int k = k0; k < std::min(k0 + TILE_SIZE, m); ++k) {
                            // Accessing b_mat (B-transposed) this way ensures sequential reads,
                            // which is optimal for cache performance.
                            sum += a_mat[i * m + k] * b_mat[j * m + k];
                        }
                        // The final result is written back to memory only once after the k-loop completes.
                        local_out_mat[i * l + j] = sum;
                    }
                }
            }
        }
    }

    // --- Gatherv logic remains unchanged ---
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
    delete[] a_mat;
    delete[] b_mat;
}





void construct_matrices_swizzle(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // For large matrices, a 1D decomposition (replicating B) is less memory-scalable
    // than a 2D decomposition. However, it's simpler and still very effective for these problem sizes.
    if (rank == 0 && (n >= LARGE_MATRIX_THRESHOLD || m >= LARGE_MATRIX_THRESHOLD || l >= LARGE_MATRIX_THRESHOLD))
    {
        //std::cout << "INFO: Using 1D block decomposition for large matrix set. Matrix B will be replicated." << std::endl;
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

void matrix_multiply_swizzle(
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

    // 3. Perform tiled matrix multiplication with the improved loop order (i0, k0, j0)
    // This order maximizes cache reuse for the `a_mat` tiles.
    for (int i0 = 0; i0 < local_n; i0 += TILE_SIZE) {
        for (int k0 = 0; k0 < m; k0 += TILE_SIZE) {
            for (int j0 = 0; j0 < l; j0 += TILE_SIZE) {
                // Multiply the tiles: A[i0..i, k0..k] * B[k0..k, j0..j] -> C[i0..i, j0..j]
                for (int i = i0; i < std::min(i0 + TILE_SIZE, local_n); ++i) {
                    for (int k = k0; k < std::min(k0 + TILE_SIZE, m); ++k) {
                        // Load a_mat[i*m+k] once and reuse it for the inner loop.
                        int a_val = a_mat[i * m + k];
                        for (int j = j0; j < std::min(j0 + TILE_SIZE, l); ++j) {
                            // Since b_mat is B-transposed, b_mat[j*m + k] accesses B[k][j].
                            // This results in sequential memory access for b_mat within this inner loop.
                            local_out_mat[i * l + j] += a_val * b_mat[j * m + k];
                        }
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