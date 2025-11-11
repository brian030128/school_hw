#include <mpi.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

// TILE_SIZE is used for on-node cache optimization for both strategies.
#define TILE_SIZE 128

// --- Global State and Strategy Selection ---

// Heuristic threshold to switch between algorithms.
const int STRATEGY_THRESHOLD = 1000;

// An enum to make the code more readable.
enum Strategy {
    UNDEFINED,
    STRATEGY_1D,
    STRATEGY_2D
};

// Static variables to hold state between function calls.
static Strategy chosen_strategy = UNDEFINED;
static int *local_a = nullptr;
static int *local_b = nullptr;
static MPI_Comm cart_comm; // Cartesian communicator for 2D strategy

// --- Forward Declarations for Helper Functions ---

// Helpers for 1D Strategy
void construct_1d(int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr);
void multiply_1d(int n, int m, int l, const int *a_mat, const int *b_mat, int *out_mat);

// Helpers for 2D Strategy
void construct_2d(int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr);
void multiply_2d(int n, int m, int l, const int *a_mat, const int *b_mat, int *out_mat);


// --- Main Interface Functions ---

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Strategy Selection Logic ---
    int grid_dim = static_cast<int>(sqrt(size));
    bool is_large_problem = (n > STRATEGY_THRESHOLD || m > STRATEGY_THRESHOLD || l > STRATEGY_THRESHOLD);
    bool is_perfect_square = (grid_dim * grid_dim == size);

    if (is_large_problem && is_perfect_square) {
        chosen_strategy = STRATEGY_2D;
    } else {
        chosen_strategy = STRATEGY_1D;
        if (rank == 0 && is_large_problem && !is_perfect_square) {
            //std::cerr << "Warning: Large matrix size detected, but the number of processes (" << size
            //          << ") is not a perfect square. Falling back to the less scalable 1D strategy.\n";
        }
    }
    
    // --- Dispatch to the correct constructor ---
    if (chosen_strategy == STRATEGY_1D) {
        construct_1d(n, m, l, a_mat, b_mat, a_mat_ptr, b_mat_ptr);
    } else { // STRATEGY_2D
        construct_2d(n, m, l, a_mat, b_mat, a_mat_ptr, b_mat_ptr);
    }
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    // --- Dispatch to the correct multiplication function ---
    if (chosen_strategy == STRATEGY_1D) {
        multiply_1d(n, m, l, a_mat, b_mat, out_mat);
    } else { // STRATEGY_2D
        multiply_2d(n, m, l, a_mat, b_mat, out_mat);
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    // The pointers a_mat and b_mat are the ones allocated in the construct functions.
    // They are correctly deallocated here.
    delete[] a_mat;
    delete[] b_mat;

    // For the 2D case, we also need to free the MPI communicator.
    if (chosen_strategy == STRATEGY_2D) {
        MPI_Comm_free(&cart_comm);
    }
}


// --- Implementation of 1D Strategy (for small matrices) ---

void construct_1d(int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_rank = n / size;
    int local_n = (rank == size - 1) ? (n - (size - 1) * rows_per_rank) : rows_per_rank;

    *a_mat_ptr = new int[local_n * m];
    *b_mat_ptr = new int[m * l];

    std::vector<int> sendcounts_a, displs_a;
    if (rank == 0) {
        sendcounts_a.resize(size);
        displs_a.resize(size);
        int current_displ = 0;
        for (int i = 0; i < size; ++i) {
            int rows = (i == size - 1) ? (n - i * rows_per_rank) : rows_per_rank;
            sendcounts_a[i] = rows * m;
            displs_a[i] = current_displ;
            current_displ += sendcounts_a[i];
        }
    }

    MPI_Scatterv(a_mat, rank == 0 ? sendcounts_a.data() : nullptr, rank == 0 ? displs_a.data() : nullptr, MPI_INT,
                 *a_mat_ptr, local_n * m, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < m * l; ++i) (*b_mat_ptr)[i] = b_mat[i];
    }
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
}

void multiply_1d(int n, int m, int l, const int *a_mat, const int *b_mat, int *out_mat) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_rank = n / size;
    int local_n = (rank == size - 1) ? (n - (size - 1) * rows_per_rank) : rows_per_rank;

    int *local_out_mat = new int[local_n * l](); // a_mat * b_mat -> out_mat

    // Tiled matrix multiplication (cache-optimized)
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
    
    std::vector<int> recvcounts_c, displs_c;
    if (rank == 0) {
        recvcounts_c.resize(size);
        displs_c.resize(size);
        int current_displ = 0;
        for (int i = 0; i < size; ++i) {
            int rows = (i == size - 1) ? (n - i * rows_per_rank) : rows_per_rank;
            recvcounts_c[i] = rows * l;
            displs_c[i] = current_displ;
            current_displ += recvcounts_c[i];
        }
    }

    MPI_Gatherv(local_out_mat, local_n * l, MPI_INT, out_mat,
                rank == 0 ? recvcounts_c.data() : nullptr, rank == 0 ? displs_c.data() : nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);

    delete[] local_out_mat;
}


// --- Implementation of 2D Strategy (for large matrices) ---

void construct_2d(int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int grid_dim = static_cast<int>(sqrt(size));
    int dims[2] = {grid_dim, grid_dim};
    int periods[2] = {1, 1}; // Toroidal wrap-around for shifts
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    // Assuming n, m, l are divisible by grid_dim for simplicity.
    // A robust implementation would handle remainders.
    int local_n = n / grid_dim;
    int local_m = m / grid_dim;
    int local_l = l / grid_dim;

    *a_mat_ptr = new int[local_n * local_m];
    *b_mat_ptr = new int[local_m * local_l];
    local_a = *a_mat_ptr; // Store in static variable for access in multiply_2d
    local_b = *b_mat_ptr;

    // Rank 0 distributes the sub-matrices to all processes
    if (rank == 0) {
        MPI_Datatype block_a, block_b;
        MPI_Type_vector(local_n, local_m, m, MPI_INT, &block_a);
        MPI_Type_commit(&block_a);
        // Note: b_mat is transposed, so its distribution logic is similar to a's
        MPI_Type_vector(local_m, local_l, l, MPI_INT, &block_b);
        MPI_Type_commit(&block_b);

        for (int i = 0; i < grid_dim; ++i) {
            for (int j = 0; j < grid_dim; ++j) {
                if (i == 0 && j == 0) continue;
                int target_rank;
                int target_coords[2] = {i, j};
                MPI_Cart_rank(cart_comm, target_coords, &target_rank);
                MPI_Send(&a_mat[(i * local_n * m) + (j * local_m)], 1, block_a, target_rank, 0, cart_comm);
                MPI_Send(&b_mat[(i * local_m * l) + (j * local_l)], 1, block_b, target_rank, 1, cart_comm);
            }
        }
        // Copy data for rank 0
        for(int i = 0; i < local_n; ++i) {
            for(int j = 0; j < local_m; ++j) {
                local_a[i * local_m + j] = a_mat[i * m + j];
            }
        }
        for(int i = 0; i < local_m; ++i) {
            for(int j = 0; j < local_l; ++j) {
                 local_b[i * local_l + j] = b_mat[i * l + j];
            }
        }
        MPI_Type_free(&block_a);
        MPI_Type_free(&block_b);
    } else {
        MPI_Recv(local_a, local_n * local_m, MPI_INT, 0, 0, cart_comm, MPI_STATUS_IGNORE);
        MPI_Recv(local_b, local_m * local_l, MPI_INT, 0, 1, cart_comm, MPI_STATUS_IGNORE);
    }

    // Cannon's Algorithm: Initial alignment
    int left, right, up, down;
    MPI_Cart_shift(cart_comm, 1, -my_row, &right, &left);
    MPI_Sendrecv_replace(local_a, local_n * local_m, MPI_INT, left, 0, right, 0, cart_comm, MPI_STATUS_IGNORE);
    
    MPI_Cart_shift(cart_comm, 0, -my_col, &down, &up);
    MPI_Sendrecv_replace(local_b, local_m * local_l, MPI_INT, up, 1, down, 1, cart_comm, MPI_STATUS_IGNORE);
}

void multiply_2d(int n, int m, int l, const int *a_mat, const int *b_mat, int *out_mat) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int grid_dim = static_cast<int>(sqrt(size));
    int local_n = n / grid_dim;
    int local_m = m / grid_dim;
    int local_l = l / grid_dim;

    int *local_c = new int[local_n * local_l]();

    int left, right, up, down;
    MPI_Cart_shift(cart_comm, 1, -1, &right, &left); // Shift left
    MPI_Cart_shift(cart_comm, 0, -1, &down, &up);   // Shift up

    // Main loop of Cannon's algorithm
    for (int stage = 0; stage < grid_dim; ++stage) {
        // Local matrix multiplication with cache tiling
        for (int i0 = 0; i0 < local_n; i0 += TILE_SIZE) {
            for (int j0 = 0; j0 < local_l; j0 += TILE_SIZE) {
                for (int k0 = 0; k0 < local_m; k0 += TILE_SIZE) {
                    for (int i = i0; i < std::min(i0 + TILE_SIZE, local_n); ++i) {
                        for (int j = j0; j < std::min(j0 + TILE_SIZE, local_l); ++j) {
                            int sum = local_c[i * local_l + j];
                            for (int k = k0; k < std::min(k0 + TILE_SIZE, local_m); ++k) {
                                // C[i,j] += A[i,k] * B[k,j]. B is NOT transposed locally here.
                                sum += local_a[i * local_m + k] * local_b[k * local_l + j];
                            }
                            local_c[i * local_l + j] = sum;
                        }
                    }
                }
            }
        }
        
        // Shift matrices for the next stage
        MPI_Sendrecv_replace(local_a, local_n * local_m, MPI_INT, left, 0, right, 0, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(local_b, local_m * local_l, MPI_INT, up, 1, down, 1, cart_comm, MPI_STATUS_IGNORE);
    }
    
    // Gather results back to rank 0
    if (rank == 0) {
        MPI_Datatype recv_block;
        MPI_Type_vector(local_n, local_l, l, MPI_INT, &recv_block);
        MPI_Type_commit(&recv_block);

        for (int i = 0; i < grid_dim; ++i) {
            for (int j = 0; j < grid_dim; ++j) {
                if (i == 0 && j == 0) continue;
                int source_rank;
                int source_coords[2] = {i, j};
                MPI_Cart_rank(cart_comm, source_coords, &source_rank);
                MPI_Recv(&out_mat[(i * local_n * l) + (j * local_l)], 1, recv_block, source_rank, 0, cart_comm, MPI_STATUS_IGNORE);
            }
        }
        // Copy local data for rank 0
        for(int i = 0; i < local_n; ++i) {
            for(int j = 0; j < local_l; ++j) {
                out_mat[i * l + j] = local_c[i * local_l + j];
            }
        }
        MPI_Type_free(&recv_block);
    } else {
        MPI_Send(local_c, local_n * local_l, MPI_INT, 0, 0, cart_comm);
    }

    delete[] local_c;
}