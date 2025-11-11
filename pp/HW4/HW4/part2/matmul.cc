#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

// TILE_SIZE is chosen for effective use of the L1d cache.
// 3 * (128 * 128) * sizeof(int) = 192 KiB, well within the 384 KiB L1d cache.
constexpr int TILE_SIZE = 128;

// Threshold to switch from a simple 1D decomposition to a more scalable 2D decomposition.
constexpr int LARGE_MATRIX_THRESHOLD = 1024;

// Forward declaration
void matrix_multiply_2d(int n, int m, int l, const int *local_a, const int *local_b, int *out_mat);

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Decide whether to use 1D or 2D decomposition.
    if (n < LARGE_MATRIX_THRESHOLD && m < LARGE_MATRIX_THRESHOLD && l < LARGE_MATRIX_THRESHOLD)
    {
        // --- 1D DECOMPOSITION (for smaller matrices) ---
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
                     *a_mat_ptr, local_n * m, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::copy(b_mat, b_mat + (m * l), *b_mat_ptr);
        }
        MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        // --- 2D DECOMPOSITION (for larger matrices) ---
        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);
        int p_rows = dims[0];
        int p_cols = dims[1];

        MPI_Comm cart_comm;
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

        int coords[2];
        MPI_Cart_coords(cart_comm, rank, 2, coords);
        int my_row = coords[0];
        int my_col = coords[1];

        // Correctly calculate local dimensions for uneven distributions.
        int base_rows = n / p_rows;
        int extra_rows = n % p_rows;
        int local_n = base_rows + (my_row < extra_rows ? 1 : 0);

        int base_cols = l / p_cols;
        int extra_cols = l % p_cols;
        int local_l = base_cols + (my_col < extra_cols ? 1 : 0);

        *a_mat_ptr = new int[local_n * m];
        *b_mat_ptr = new int[local_l * m];

        MPI_Comm row_comm, col_comm;
        MPI_Comm_split(cart_comm, my_row, my_col, &row_comm);
        MPI_Comm_split(cart_comm, my_col, my_row, &col_comm);

        // Distribute Matrix A using Scatterv to the first column, then Bcast across rows.
        if (my_col == 0)
        {
            std::vector<int> sendcounts_a(p_rows);
            std::vector<int> displs_a(p_rows);
            if (rank == 0)
            {
                int current_displ = 0;
                for (int i = 0; i < p_rows; ++i)
                {
                    int rows_for_this_rank = base_rows + (i < extra_rows ? 1 : 0);
                    sendcounts_a[i] = rows_for_this_rank * m;
                    displs_a[i] = current_displ;
                    current_displ += sendcounts_a[i];
                }
            }
            MPI_Scatterv(a_mat, sendcounts_a.data(), displs_a.data(), MPI_INT,
                         *a_mat_ptr, local_n * m, MPI_INT, 0, col_comm);
        }
        MPI_Bcast(*a_mat_ptr, local_n * m, MPI_INT, 0, row_comm);

        // Distribute Matrix B_T using Scatterv to the first row, then Bcast down columns.
        if (my_row == 0)
        {
            std::vector<int> sendcounts_b(p_cols);
            std::vector<int> displs_b(p_cols);
            if (rank == 0)
            {
                int current_displ = 0;
                for (int i = 0; i < p_cols; ++i)
                {
                    int cols_for_this_rank = base_cols + (i < extra_cols ? 1 : 0);
                    sendcounts_b[i] = cols_for_this_rank * m;
                    displs_b[i] = current_displ;
                    current_displ += sendcounts_b[i];
                }
            }
            MPI_Scatterv(b_mat, sendcounts_b.data(), displs_b.data(), MPI_INT,
                         *b_mat_ptr, local_l * m, MPI_INT, 0, row_comm);
        }
        MPI_Bcast(*b_mat_ptr, local_l * m, MPI_INT, 0, col_comm);

        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);
        MPI_Comm_free(&cart_comm);
    }
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    if (n < LARGE_MATRIX_THRESHOLD && m < LARGE_MATRIX_THRESHOLD && l < LARGE_MATRIX_THRESHOLD)
    {
        // --- 1D Tiled Multiplication ---
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
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
    else
    {
        matrix_multiply_2d(n, m, l, a_mat, b_mat, out_mat);
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    delete[] a_mat;
    delete[] b_mat;
}

void matrix_multiply_2d(int n, int m, int l, const int *local_a, const int *local_b, int *out_mat)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int p_rows = dims[0];
    int p_cols = dims[1];

    MPI_Comm cart_comm;
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    // Recalculate local dimensions exactly as in construct_matrices.
    int local_n = (n / p_rows) + (my_row < (n % p_rows) ? 1 : 0);
    int local_l = (l / p_cols) + (my_col < (l % p_cols) ? 1 : 0);

    std::vector<int> local_c(local_n * local_l, 0);

    // Perform local tiled matrix multiplication.
    for (int i0 = 0; i0 < local_n; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < local_l; j0 += TILE_SIZE) {
            for (int k0 = 0; k0 < m; k0 += TILE_SIZE) {
                for (int i = i0; i < std::min(i0 + TILE_SIZE, local_n); ++i) {
                    for (int j = j0; j < std::min(j0 + TILE_SIZE, local_l); ++j) {
                        int sum = local_c[i * local_l + j];
                        for (int k = k0; k < std::min(k0 + TILE_SIZE, m); ++k) {
                            sum += local_a[i * m + k] * local_b[j * m + k];
                        }
                        local_c[i * local_l + j] = sum;
                    }
                }
            }
        }
    }

    // Gather all local results onto rank 0.
    if (rank == 0)
    {
        // Rank 0 must calculate the size and displacement for every process's result block.
        std::vector<int> rows_per_rank(p_rows);
        std::vector<int> row_displs(p_rows + 1, 0);
        std::vector<int> cols_per_rank(p_cols);
        std::vector<int> col_displs(p_cols + 1, 0);

        for (int i = 0; i < p_rows; ++i) {
            rows_per_rank[i] = (n / p_rows) + (i < (n % p_rows) ? 1 : 0);
            row_displs[i+1] = row_displs[i] + rows_per_rank[i];
        }
        for (int i = 0; i < p_cols; ++i) {
            cols_per_rank[i] = (l / p_cols) + (i < (l % p_cols) ? 1 : 0);
            col_displs[i+1] = col_displs[i] + cols_per_rank[i];
        }
        
        // Copy rank 0's own data first.
        int row_offset = row_displs[0];
        int col_offset = col_displs[0];
        for (int i = 0; i < local_n; ++i) {
            for (int j = 0; j < local_l; ++j) {
                out_mat[(row_offset + i) * l + (col_offset + j)] = local_c[i * local_l + j];
            }
        }
        
        // Receive data from all other processes.
        for (int src_rank = 1; src_rank < size; ++src_rank)
        {
            int src_coords[2];
            MPI_Cart_coords(cart_comm, src_rank, 2, src_coords);
            int proc_row = src_coords[0];
            int proc_col = src_coords[1];

            int recv_n = rows_per_rank[proc_row];
            int recv_l = cols_per_rank[proc_col];
            
            std::vector<int> result_buffer(recv_n * recv_l);
            MPI_Recv(result_buffer.data(), result_buffer.size(), MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            row_offset = row_displs[proc_row];
            col_offset = col_displs[proc_col];
            for (int i = 0; i < recv_n; ++i) {
                for (int j = 0; j < recv_l; ++j) {
                    out_mat[(row_offset + i) * l + (col_offset + j)] = result_buffer[i * recv_l + j];
                }
            }
        }
    }
    else
    {
        MPI_Send(local_c.data(), local_c.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Comm_free(&cart_comm);
}