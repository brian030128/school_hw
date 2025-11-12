#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm> // For std::min and std::copy

// TILE_SIZE is chosen to ensure the working set of three tiles (from A, B, and C)
// fits within the L1d cache.
// 3 * (128 * 128) * sizeof(int) = 192 KiB.
constexpr int TILE_SIZE = 128;

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    int rows_per_rank = n / size;
    int local_n = (rank == size - 1) ? (n - (size - 1) * rows_per_rank) : rows_per_rank;

    std::vector<int> local_out_mat(local_n * l, 0);


    for (int i0 = 0; i0 < local_n; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < l; j0 += TILE_SIZE) {
            // The two outer loops select a single tile from the C matrix (local_out_mat).
            // This tile will be held in cache while it is being computed.
            for (int k0 = 0; k0 < m; k0 += TILE_SIZE) {
                // This innermost tiling loop iterates through the necessary A and B tiles
                // to fully compute the selected C tile.
                for (int i = i0; i < std::min(i0 + TILE_SIZE, local_n); ++i) {
                    for (int j = j0; j < std::min(j0 + TILE_SIZE, l); ++j) {
                        int sum = local_out_mat[i * l + j];
                        for (int k = k0; k < std::min(k0 + TILE_SIZE, m); ++k) {
                            sum += a_mat[i * m + k] * b_mat[j * m + k];
                        }
                        local_out_mat[i * l + j] = sum;
                    }
                }
            }
        }
    }

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



