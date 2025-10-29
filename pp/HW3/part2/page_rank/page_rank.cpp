#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{
    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;

    // Allocate temporary arrays.
    // score_old holds the scores from the previous iteration.
    // We will compute the new scores into the solution array directly.
    double *score_old = new double[nnodes];

    // Initialize scores to a uniform probability.
    #pragma omp parallel for
    for (int i = 0; i < nnodes; ++i) {
        score_old[i] = equal_prob;
    }

    bool converged = false;
    while (!converged) {

        // Calculate the sum of scores for all dangling nodes (nodes with no outgoing edges).
        // This is a global value used in the score update for all nodes.
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for (int i = 0; i < nnodes; ++i) {
            if (outgoing_size(g, i) == 0) {
                dangling_sum += score_old[i];
            }
        }

        // The core PageRank calculation loop.
        // This loop is parallelized, with each thread handling a subset of the nodes.
        #pragma omp parallel for
        for (int i = 0; i < nnodes; ++i) {
            // Start by calculating the sum of contributions from incoming neighbors.
            double sum_incoming = 0.0;
            const Vertex* start = incoming_begin(g, i);
            const Vertex* end = incoming_end(g, i);
            for (const Vertex* v_ptr = start; v_ptr != end; ++v_ptr) {
                int neighbor_idx = *v_ptr;
                sum_incoming += score_old[neighbor_idx] / outgoing_size(g, neighbor_idx);
            }

            // Apply the PageRank formula:
            // 1. Damped sum from incoming neighbors.
            double new_score = (damping * sum_incoming);
            // 2. Random jump probability.
            new_score += (1.0 - damping) / nnodes;
            // 3. Contribution from dangling nodes, distributed evenly.
            new_score += damping * dangling_sum / nnodes;

            solution[i] = new_score;
        }

        // Check for convergence by summing the absolute differences between
        // the new scores (in solution) and the old scores.
        double global_diff = 0.0;
        #pragma omp parallel for reduction(+:global_diff)
        for (int i = 0; i < nnodes; ++i) {
            global_diff += std::abs(solution[i] - score_old[i]);
        }
        converged = (global_diff < convergence);

        // Prepare for the next iteration: copy new scores into score_old.
        #pragma omp parallel for
        for (int i = 0; i < nnodes; ++i) {
            score_old[i] = solution[i];
        }
    }

    // Free the allocated temporary array.
    delete[] score_old;
}