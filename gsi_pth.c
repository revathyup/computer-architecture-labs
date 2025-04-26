#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdatomic.h>

#include "gs_interface.h"

const int gsi_is_parallel = 1;

typedef struct {
    int thread_id;
    pthread_t thread;
    double error;
    char padding[64 - sizeof(int) - sizeof(pthread_t) - sizeof(double)]; // Padding to prevent false sharing
} thread_info_t;

#define DEBUG 0
#define dprintf(...) if (DEBUG) fprintf(stderr, __VA_ARGS__)

thread_info_t *threads = NULL;
static double global_error;

// Synchronization variables
static _Atomic int *progress_counter;
static pthread_barrier_t iter_barrier;

void gsi_init()
{
    gs_verbose_printf("\t**** Initializing environment ****\n");

    threads = (thread_info_t *)malloc(gs_nthreads * sizeof(thread_info_t));
    if (!threads) {
        fprintf(stderr, "Failed to allocate thread information.\n");
        exit(EXIT_FAILURE);
    }

    global_error = gs_tolerance + 1;  // Ensure we start with error > tolerance

    // Initialize progress counter
    progress_counter = (_Atomic int *)malloc(gs_size * sizeof(_Atomic int));
    if (!progress_counter) {
        fprintf(stderr, "Failed to allocate progress counter.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < gs_size; i++)
        atomic_init(&progress_counter[i], -1);

    pthread_barrier_init(&iter_barrier, NULL, gs_nthreads);
}

void gsi_finish()
{
    gs_verbose_printf("\t**** Cleaning environment ****\n");

    free(progress_counter);
    pthread_barrier_destroy(&iter_barrier);
    free(threads);
}

static void thread_sweep(int tid, int iter, int lbound, int rbound)
{
    threads[tid].error = 0.0;

    for (int row = 1; row < gs_size - 1; row++) {
        // Wait for previous row to complete in this iteration
        if (row > 1) {
            while (atomic_load(&progress_counter[row-1]) < iter) {
                // Brief pause to reduce contention
                for (volatile int i = 0; i < 100; i++);
            }
        }

        // Process current row
        for (int col = lbound; col < rbound; col++) {
            double new_value = 0.25 * (
                gs_matrix[GS_INDEX(row + 1, col)] +
                gs_matrix[GS_INDEX(row - 1, col)] +
                gs_matrix[GS_INDEX(row, col + 1)] +
                gs_matrix[GS_INDEX(row, col - 1)]);
            threads[tid].error += fabs(gs_matrix[GS_INDEX(row, col)] - new_value);
            gs_matrix[GS_INDEX(row, col)] = new_value;
        }

        // Mark row as complete for this iteration
        atomic_store(&progress_counter[row], iter);
    }
}

static void *thread_compute(void *_self)
{
    thread_info_t *self = (thread_info_t *)_self;
    const int tid = self->thread_id;

    // Calculate column bounds for this thread
    const int chunk_size = (gs_size - 2 + gs_nthreads - 1) / gs_nthreads;
    const int lbound = 1 + tid * chunk_size;
    const int rbound = (tid == gs_nthreads - 1) ? gs_size - 1 : lbound + chunk_size;

    for (int iter = 0; iter < gs_iterations && global_error > gs_tolerance; iter++) {
        thread_sweep(tid, iter, lbound, rbound);

        // Barrier: Wait for all threads to finish sweep
        pthread_barrier_wait(&iter_barrier);

        // Thread 0 computes global error
        if (tid == 0) {
            global_error = 0.0;
            for (int t = 0; t < gs_nthreads; t++) {
                global_error += threads[t].error;
            }
        }

        // Barrier: Wait for global error to be updated
        pthread_barrier_wait(&iter_barrier);
    }

    return NULL;
}

void gsi_calculate()
{
    // Initialize all thread errors to 0
    for (int t = 0; t < gs_nthreads; t++) {
        threads[t].thread_id = t;
        threads[t].error = 0.0;
    }

    // Create threads
    for (int t = 0; t < gs_nthreads; t++) {
        if (pthread_create(&threads[t].thread, NULL, thread_compute, &threads[t])) {
            perror("pthread_create failed");
            exit(EXIT_FAILURE);
        }
    }

    // Join threads
    for (int t = 0; t < gs_nthreads; t++) {
        if (pthread_join(threads[t].thread, NULL)) {
            perror("pthread_join failed");
            exit(EXIT_FAILURE);
        }
    }

    if (global_error <= gs_tolerance) {
        printf("Solution converged!\n");
    } else {
        printf("Reached maximum number of iterations. Solution did NOT converge.\n");
        printf("Note: This is normal if you are using the default settings.\n");
    }
}