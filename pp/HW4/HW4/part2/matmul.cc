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

    // Decide whether to use 1D or 2D decomposition based on matrix size.
    if (n < LARGE_MATRIX_THRESHOLD && m < LARGE_MATRIX_THRESHOLD && l < LARGE_MATRIX_THRESHOLD)
    {
        // --- 1D DECOMPOSITION (for smaller matrices) ---
        int rows_per_rank = n / size;
        int local_n = (rank == size - 1) ? (n - (size - 1) * rows_per_rank) : rows_per_rank;

        *a_mat_ptr = new int[local_n * m];
        *b_mat_ptr = new int[m * l]; // Each process gets a full copy of B.

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
        // 1. Create a 2D Cartesian process grid.
        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);
        int p_rows = dims[0];
        int p_cols = dims[1];

        MPI_Comm cart_comm;
        int periods[2] = {0, 0}; // Not periodic
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

        int coords[2];
        MPI_Cart_coords(cart_comm, rank, 2, coords);
        int my_row = coords[0];
        int my_col = coords[1];

        // 2. Determine local matrix dimensions for each process.
        int local_n = n / p_rows;
        int local_l = l / p_cols;

        // 3. Allocate memory for local sub-matrices.
        *a_mat_ptr = new int[local_n * m];      // Each process gets a horizontal strip of A.
        *b_mat_ptr = new int[local_l * m];      // Each process gets a horizontal strip of B_T.

        // Create communicators for rows and columns of the process grid.
        MPI_Comm row_comm, col_comm;
        MPI_Comm_split(cart_comm, my_row, my_col, &row_comm);
        MPI_Comm_split(cart_comm, my_col, my_row, &col_comm);
        
        // 4. Distribute Matrix A.
        if (my_col == 0) { // Processes in the first column receive a strip from rank 0.
            std::vector<int> sendcounts_a(p_rows);
            std::vector<int> displs_a(p_rows);
            if (rank == 0) { // Rank 0 is also (0,0), it prepares scatter info.
                 for(int i=0; i<p_rows; ++i) {
                    sendcounts_a[i] = local_n * m;
                    displs_a[i] = i * local_n * m;
                 }
            }
             MPI_Scatterv(a_mat, sendcounts_a.data(), displs_a.data(), MPI_INT,
                         *a_mat_ptr, local_n * m, MPI_INT, 0, col_comm);
        }
        // Broadcast the strip of A across the process row.
        MPI_Bcast(*a_mat_ptr, local_n * m, MPI_INT, 0, row_comm);

        // 5. Distribute Matrix B (which is pre-transposed, B_T).
        if (my_row == 0) { // Processes in the first row receive a strip from rank 0.
            std::vector<int> sendcounts_b(p_cols);
            std::vector<int> displs_b(p_cols);
            if(rank == 0){ // Rank 0 prepares scatter info.
                for(int i=0; i<p_cols; ++i) {
                    sendcounts_b[i] = local_l * m;
                    displs_b[i] = i * local_l * m;
                }
            }
            MPI_Scatterv(b_mat, sendcounts_b.data(), displs_b.data(), MPI_INT,
                         *b_mat_ptr, local_l * m, MPI_INT, 0, row_comm);
        }
        // Broadcast the strip of B_T down the process column.
        MPI_Bcast(*b_mat_ptr, local_l * m, MPI_INT, 0, col_comm);

        // 6. Clean up communicators.
        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);
        MPI_Comm_free(&cart_comm);
    }
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    // Dispatch to the correct implementation based on matrix size.
    // The `a_mat` and `b_mat` pointers now always point to valid local data.
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
        
        int rank_zero = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_zero);
        std::vector<int> recvcounts_c(size);
        std::vector<int> displs_c(size);

        if (rank_zero == 0)
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
        // --- 2D Tiled Multiplication ---
        matrix_multiply_2d(n, m, l, a_mat, b_mat, out_mat);
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    // This check is good practice, although with the corrected logic,
    // these pointers should not be null if allocation was successful.
    if (a_mat != nullptr) delete[] a_mat;
    if (b_mat != nullptr) delete[] b_mat;
}

// Performs local computation for the 2D case and gathers the final result.
void matrix_multiply_2d(int n, int m, int l, const int *local_a, const int *local_b, int *out_mat)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Recreate Cartesian grid to get process coordinates.
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int p_rows = dims[0];
    int p_cols = dims[1];

    MPI_Comm cart_comm;
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    // 2. Determine local matrix dimensions again.
    int local_n = n / p_rows;
    int local_l = l / p_cols;

    // 3. Allocate and initialize the local result matrix C.
    std::vector<int> local_c(local_n * local_l, 0);

    // 4. Perform local tiled matrix multiplication.
    for (int i0 = 0; i0 < local_n; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < local_l; j0 += TILE_SIZE) {
            for (int k0 = 0; k0 < m; k0 += TILE_SIZE) {
                for (int i = i0; i < std::min(i0 + TILE_SIZE, local_n); ++i) {
                    for (int j = j0; j < std::min(j0 + TILE_SIZE, local_l); ++j) {
                        int sum = local_c[i * local_l + j];
                        for (int k = k0; k < std::min(k0 + TILE_SIZE, m); ++k) {
                            // local_a is a local_n x m matrix.
                            // local_b is a local_l x m matrix (a piece of B_T).
                            sum += local_a[i * m + k] * local_b[j * m + k];
                        }
                        local_c[i * local_l + j] = sum;
                    }
                }
            }
        }
    }

    // 5. Gather all local results into the final output matrix on rank 0.
    // We use a simple Send/Recv loop for clarity.
    if (rank == 0) {
        std::vector<int> result_buffer(local_n * local_l);
        for (int proc_row = 0; proc_row < p_rows; ++proc_row) {
            for (int proc_col = 0; proc_col < p_cols; ++proc_col) {
                int src_rank;
                int src_coords[2] = {proc_row, proc_col};
                MPI_Cart_rank(cart_comm, src_coords, &src_rank);
                
                // Decide where to get the data from.
                const int* data_ptr = (src_rank == 0) ? local_c.data() : result_buffer.data();

                if (src_rank != 0) {
                    MPI_Recv(result_buffer.data(), local_n * local_l, MPI_INT, src_rank, 0, cart_comm, MPI_STATUS_IGNORE);
                }

                // Copy the received block into the final output matrix.
                for (int i = 0; i < local_n; ++i) {
                    for (int j = 0; j < local_l; ++j) {
                        out_mat[(proc_row * local_n + i) * l + (proc_col * local_l + j)] = data_ptr[i * local_l + j];
                    }
                }
            }
        }
    } else {
        MPI_Send(local_c.data(), local_n * local_l, MPI_INT, 0, 0, cart_comm);
    }
    
    MPI_Comm_free(&cart_comm);
}