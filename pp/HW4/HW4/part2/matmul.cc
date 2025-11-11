#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

// TILE_SIZE is chosen for effective use of the L1d cache.
// 3 * (128 * 128) * sizeof(int) = 192 KiB, well within the 384 KiB L1d cache.
constexpr int TILE_SIZE = 128;
constexpr int LARGE_MATRIX_THRESHOLD = 1024;

// Forward declaration for the 2D multiplication function
void matrix_multiply_2d(int n, int m, int l, const int* a_mat, const int* b_mat, int* out_mat, int* local_a, int* local_b);

void construct_matrices(
    int n, int m, int l, const int* a_mat, const int* b_mat, int** a_mat_ptr, int** b_mat_ptr)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Use 2D decomposition for large matrices, otherwise use the simpler 1D approach.
    if (n >= LARGE_MATRIX_THRESHOLD || m >= LARGE_MATRIX_THRESHOLD || l >= LARGE_MATRIX_THRESHOLD) {
        // For the 2D case, the distribution happens inside the multiply function.
        // We only need to allocate placeholder pointers.
        *a_mat_ptr = nullptr;
        *b_mat_ptr = nullptr;
    } else {
        // 1D Decomposition (original logic)
        int rows_per_rank = n / size;
        int local_n = (rank == size - 1) ? (n - (size - 1) * rows_per_rank) : rows_per_rank;

        *a_mat_ptr = new int[local_n * m];
        *b_mat_ptr = new int[m * l];

        std::vector<int> sendcounts_a(size);
        std::vector<int> displs_a(size);

        if (rank == 0) {
            int current_displ = 0;
            for (int i = 0; i < size; ++i) {
                int rows_for_this_rank = (i == size - 1) ? (n - i * rows_per_rank) : rows_per_rank;
                sendcounts_a[i] = rows_for_this_rank * m;
                displs_a[i] = current_displ;
                current_displ += sendcounts_a[i];
            }
        }

        MPI_Scatterv(a_mat, sendcounts_a.data(), displs_a.data(), MPI_INT,
                     *a_mat_ptr, local_n * m, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::copy(b_mat, b_mat + (m * l), *b_mat_ptr);
        }
        MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void matrix_multiply(
    const int n, const int m, const int l, const int* a_mat, const int* b_mat, int* out_mat)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (n >= LARGE_MATRIX_THRESHOLD || m >= LARGE_MATRIX_THRESHOLD || l >= LARGE_MATRIX_THRESHOLD) {
        matrix_multiply_2d(n, m, l, a_mat, b_mat, out_mat, nullptr, nullptr);
    } else {
        // 1D Tiled Multiplication (original logic)
        int rows_per_rank = n / size;
        int local_n = (rank == size - 1) ? (n - (size - 1) * rows_per_rank) : rows_per_rank;

        std::vector<int> local_out_mat(local_n * l, 0);

        for (int i0 = 0; i0 < local_n; i0 += TILE_SIZE) {
            for (int j0 = 0; j0 < l; j0 += TILE_SIZE) {
                for (int k0 = 0; k0 < m; k0 += TILE_SIZE) {
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

        if (rank == 0) {
            int current_displ = 0;
            for (int i = 0; i < size; ++i) {
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
}

void destruct_matrices(int* a_mat, int* b_mat)
{
    // The new 2D implementation doesn't allocate in construct_matrices, so check for null.
    if (a_mat != nullptr) delete[] a_mat;
    if (b_mat != nullptr) delete[] b_mat;
}


// 2D Scalable Matrix Multiplication
void matrix_multiply_2d(int n, int m, int l, const int* a_mat, const int* b_mat, int* out_mat, int* local_a, int* local_b)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Create a 2D Cartesian communicator.
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int p_rows = dims[0];
    int p_cols = dims[1];

    MPI_Comm cart_comm;
    int periods[2] = {1, 1}; // Periodic for shifts
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    // 2. Determine local matrix dimensions.
    int local_n = n / p_rows;
    int local_l = l / p_cols;

    std::vector<int> local_c(local_n * local_l, 0);
    std::vector<int> temp_a(local_n * m);
    std::vector<int> temp_b(m * local_l);

    // 3. Distribute matrices A and B.
    if (rank == 0) {
        // Rank 0 sends slices of A and B to the first process in each row/column.
        for (int i = 0; i < p_rows; ++i) {
            for (int j = 0; j < p_cols; ++j) {
                if (i == 0 && j == 0) continue; // Skip self

                // Send a horizontal strip of A to the first process of each row.
                if (j == 0) {
                    int target_rank;
                    int dest_coords[2] = {i, j};
                    MPI_Cart_rank(cart_comm, dest_coords, &target_rank);
                    for (int row_idx = 0; row_idx < local_n; ++row_idx) {
                        MPI_Send(&a_mat[(i * local_n + row_idx) * m], m, MPI_INT, target_rank, 0, cart_comm);
                    }
                }
                // Send a vertical strip of B to the first process of each col.
                if (i == 0) {
                    int target_rank;
                    int dest_coords[2] = {i, j};
                    MPI_Cart_rank(cart_comm, dest_coords, &target_rank);
                    std::vector<int> b_slice(m * local_l);
                    for(int row = 0; row < m; ++row) {
                        for(int col = 0; col < local_l; ++col) {
                            b_slice[row * local_l + col] = b_mat[(j * local_l + col) * m + row]; // Transposed access
                        }
                    }
                    MPI_Send(b_slice.data(), m * local_l, MPI_INT, target_rank, 1, cart_comm);
                }
            }
        }
        // Rank 0's local data
        for (int i = 0; i < local_n; ++i) std::copy(&a_mat[i * m], &a_mat[i * m] + m, &temp_a[i * m]);
        for(int row = 0; row < m; ++row) {
            for(int col = 0; col < local_l; ++col) {
                temp_b[row * local_l + col] = b_mat[col * m + row]; // Transposed access
            }
        }

    } else {
        // Receive initial data if on the first row or column.
        if (my_col == 0) {
            for (int i = 0; i < local_n; ++i) {
                MPI_Recv(&temp_a[i * m], m, MPI_INT, 0, 0, cart_comm, MPI_STATUS_IGNORE);
            }
        }
        if (my_row == 0) {
            MPI_Recv(temp_b.data(), m * local_l, MPI_INT, 0, 1, cart_comm, MPI_STATUS_IGNORE);
        }
    }

    // 4. Broadcast A across rows and B down columns.
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(cart_comm, my_col, my_row, &col_comm);

    MPI_Bcast(temp_a.data(), local_n * m, MPI_INT, 0, row_comm);
    MPI_Bcast(temp_b.data(), m * local_l, MPI_INT, 0, col_comm);

    // 5. Perform local tiled matrix multiplication.
    for (int i0 = 0; i0 < local_n; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < local_l; j0 += TILE_SIZE) {
            for (int k0 = 0; k0 < m; k0 += TILE_SIZE) {
                for (int i = i0; i < std::min(i0 + TILE_SIZE, local_n); ++i) {
                    for (int j = j0; j < std::min(j0 + TILE_SIZE, local_l); ++j) {
                        int sum = local_c[i * local_l + j];
                        for (int k = k0; k < std::min(k0 + TILE_SIZE, m); ++k) {
                           sum += temp_a[i * m + k] * temp_b[k * local_l + j];
                        }
                        local_c[i * local_l + j] = sum;
                    }
                }
            }
        }
    }

    // 6. Gather the results on rank 0.
    if (rank == 0) {
        std::vector<int> result_buffer(local_n * local_l);
        for (int proc_row = 0; proc_row < p_rows; ++proc_row) {
            for (int proc_col = 0; proc_col < p_cols; ++proc_col) {
                int src_rank;
                int src_coords[2] = {proc_row, proc_col};
                MPI_Cart_rank(cart_comm, src_coords, &src_rank);

                if (src_rank == 0) {
                     std::copy(local_c.begin(), local_c.end(), result_buffer.begin());
                } else {
                    MPI_Recv(result_buffer.data(), local_n * local_l, MPI_INT, src_rank, 3, cart_comm, MPI_STATUS_IGNORE);
                }

                // Copy the received block into the final output matrix.
                for (int i = 0; i < local_n; ++i) {
                    for (int j = 0; j < local_l; ++j) {
                        out_mat[(proc_row * local_n + i) * l + (proc_col * local_l + j)] = result_buffer[i * local_l + j];
                    }
                }
            }
        }
    } else {
        MPI_Send(local_c.data(), local_n * local_l, MPI_INT, 0, 3, cart_comm);
    }
    
    // Cleanup
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
}