#include "bfs.h"

#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = distances[node] + 1;
                int index = new_frontier->count++;
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
}

void top_down_step(Graph graph, VertexSet* frontier, VertexSet* new_frontier, int* distances) {
    // Parallelize the loop over the current frontier.
    // Each thread will process a subset of the vertices in the frontier.
    #pragma omp parallel for
    for (int i = 0; i < frontier->count; i++) {
        Vertex node = frontier->vertices[i];
        int dist = distances[node];

        const Vertex* start = outgoing_begin(graph, node);
        const Vertex* end = outgoing_end(graph, node);

        // Iterate over the neighbors of the current node.
        for (const Vertex* neighbor = start; neighbor != end; ++neighbor) {
            // Use an atomic read to prevent race conditions on the distances array.
            int neighbor_dist;
            #pragma omp atomic read
            neighbor_dist = distances[*neighbor];

            if (neighbor_dist == NOT_VISITED_MARKER) {
                distances[*neighbor] = dist + 1;

                // Use a critical section to safely add the new neighbor to the next frontier.
                // This prevents multiple threads from trying to write to the same spot simultaneously.
                #pragma omp critical
                {
                    new_frontier->vertices[new_frontier->count] = *neighbor;
                    new_frontier->count++;
                }
            }
        }
    }
}

// Implements the top-down parallel BFS.
void bfs_top_down(Graph graph, solution* sol) {
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet* frontier = &list1;
    VertexSet* new_frontier = &list2;

    // Initialize all node distances to NOT_VISITED_MARKER in parallel.
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // Setup the frontier with the root node.
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // Continue as long as there are nodes in the frontier to process.
    while (frontier->count != 0) {
        vertex_set_clear(new_frontier);
        top_down_step(graph, frontier, new_frontier, sol->distances);

        // Swap frontier pointers for the next iteration.
        VertexSet* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // Free the memory used by the vertex sets.
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
