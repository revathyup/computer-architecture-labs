/**
 * Parallel Gauss-Seidel implementation using pthreads.
 *
 * Course: Advanced Computer Architecture, Uppsala University
 * Course Part: Lab assignment 3
 */

 #include <pthread.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <stdatomic.h>                   // 1>For atomic operations
 
 #include "gs_interface.h"
 
 /* 2> Define this to enable debug printing */
 #define DEBUG 0
 
 #if DEBUG
 #define dprintf(...) gs_verbose_printf(__VA_ARGS__)
 #else
 #define dprintf(...) /* Don't print anything */
 #endif
 
 const int gsi_is_parallel = 1;
 
 /**
  * Thread information structure
  * Contains all the information needed by the worker threads
  */
 typedef struct {
     int thread_id;        /* Thread ID */
     pthread_t thread;     /* pthread handle */
     double error;         /* Local error for this thread */
     _Atomic int row_progress; /* 3> Progress counter for row synchronization */
     /* 4> Padding to prevent false sharing - ensure struct occupies a full cache line */
     char padding[64 - (sizeof(int) + sizeof(pthread_t) + sizeof(double) + sizeof(_Atomic int)) % 64];
 } __attribute__((aligned(64))) thread_info_t;
 
 /* Global variables */
 static thread_info_t *threads = NULL;
 static double global_error;
 static pthread_barrier_t iter_barrier;  // 5>barrier for thread synchronization
 static int final_iteration;            // 6>final iteration count
 
 /**
  * Initialize the thread information structures and other shared data.
  */
 void gsi_init() {
     gs_verbose_printf("\t****  Initializing parallel environment ****\n");
 
     /* 7> Allocate and initialize thread info structures with cache line alignment to avoid false sharing */
     threads = (thread_info_t *)aligned_alloc(64, gs_nthreads * sizeof(thread_info_t));
     if (!threads) {
         fprintf(stderr, "Failed to allocate thread info\n");
         exit(EXIT_FAILURE);
     }
 
     global_error = gs_tolerance + 1;
     /* 8> Initialize global variables */
     pthread_barrier_init(&iter_barrier, NULL, gs_nthreads);
     final_iteration = gs_iterations;
 
     /* 9> Initialize thread-specific data */
     for (int i = 0; i < gs_nthreads; i++) {
         threads[i].thread_id = i;
         threads[i].error = 0.0;
         atomic_init(&threads[i].row_progress, 0);
     }
     
     dprintf("\t****  Parallel environment initialized with %d threads ****\n", gs_nthreads);
 }
 
 /**
  * Clean up the thread information structures and other shared data.
  */
 void gsi_finish() {
     gs_verbose_printf("\t****  Cleaning parallel environment ****\n");
     /* 10> destroyed or cleaned*/
     pthread_barrier_destroy(&iter_barrier);
     free(threads);
 }
 
 /**
  * Performs a sweep of the Gauss-Seidel algorithm for a single thread.
  * Each thread works on its own vertical chunk of the matrix.
  * Threads synchronize using row_progress counters to ensure correct data dependencies.
  */
 static void thread_sweep(int tid, int start_col, int end_col) {
     double local_error = 0.0;
     
     /* We're iterating over interior points only, so we start at row 1 and end at (gs_size-2) */
     for (int i = 1; i < gs_size - 1; i++) {
         /*  11> Wait for the thread to the left to finish this row before we start */
         if (tid > 0) {
             while (atomic_load(&threads[tid - 1].row_progress) < i) {
                 /* Short busy wait to reduce contention */
                 for (volatile int k = 0; k < 10; k++);
             }
         }
         
         /* Update each point in our assigned area */
         for (int j = start_col; j < end_col; j++) {
             /* Calculate new value using the standard stencil */
             double new_value = 0.25 * (
                 gs_matrix[GS_INDEX(i + 1, j)] +  /* Below (old value) */
                 gs_matrix[GS_INDEX(i - 1, j)] +  /* Above (old value) */
                 gs_matrix[GS_INDEX(i, j + 1)] +  /* Right (old value) */
                 gs_matrix[GS_INDEX(i, j - 1)]    /* Left (new value if updated) */
             );
             
             /* Calculate local error */
             local_error += fabs(gs_matrix[GS_INDEX(i, j)] - new_value);
             
             /* 11> Update the matrix in-place with the new value */
             gs_matrix[GS_INDEX(i, j)] = new_value;
         }
         
         /* 12> Signal that we've completed processing this row */
         atomic_store(&threads[tid].row_progress, i);
     }
     
     /* Store the local error in the thread's structure */
     threads[tid].error = local_error;
 }
 
 /**
  * Main computation function for each thread.
  * This function is executed by each worker thread.
  */
 static void *thread_compute(void *_self) {
     thread_info_t *self = (thread_info_t *)_self;
     int tid = self->thread_id;
     
     /* 13> Calculate the column range for this thread */
     int interior_size = gs_size - 2;  /* number of interior points in each dimension */
     int points_per_thread = interior_size / gs_nthreads;
     int start_col = 1 + tid * points_per_thread;
     int end_col = (tid == gs_nthreads - 1) ? gs_size - 1 : start_col + points_per_thread;
     
     dprintf("Thread %d working on columns %d to %d\n", tid, start_col, end_col);
     
     /* Main iteration loop */
     for (int iter = 0; iter < gs_iterations; iter++) {
         /* 14> Reset progress indicators before starting a new iteration */
         atomic_store(&threads[tid].row_progress, 0);
         
         /* 15> Synchronize all threads before starting a new iteration */
         pthread_barrier_wait(&iter_barrier);
         
         /* Process this thread's part of the matrix */
         thread_sweep(tid, start_col, end_col);
         
         /*16>  Wait for all threads to finish their sweep before accumulating errors */
         pthread_barrier_wait(&iter_barrier);
         
         /*17> Thread 0 computes the global error and checks for convergence */
         if (tid == 0) {
             global_error = 0.0;
             for (int t = 0; t < gs_nthreads; t++) {
                 global_error += threads[t].error;
             }
             
             dprintf("Iteration: %i, Error: %f\n", iter, global_error);
             
             /* Check for convergence */
             if (global_error <= gs_tolerance) {
                 final_iteration = iter + 1;
                 /* We could break here, but we'll let all threads run for the specified iterations */
             }
         }
         
         /* Wait for error calculation before potentially starting next iteration */
         pthread_barrier_wait(&iter_barrier);
     }
     
     return NULL;
 }
 
 /**
  * Main entry point for the Gauss-Seidel calculation.
  * Creates threads, starts the calculation, and waits for threads to finish.
  */
 void gsi_calculate() {
     gs_verbose_printf("\t****  Starting parallel Gauss-Seidel calculation ****\n");
     
     /* Create and start the worker threads */
     for (int t = 0; t < gs_nthreads; t++) {
         if (pthread_create(&threads[t].thread, NULL, thread_compute, &threads[t]) != 0) {
             fprintf(stderr, "Error creating thread %d\n", t);
             exit(EXIT_FAILURE);
         }
     }
     
     /* Wait for all threads to complete */
     for (int t = 0; t < gs_nthreads; t++) {
         pthread_join(threads[t].thread, NULL);
     }
     
     /* Print convergence information */
     if (global_error <= gs_tolerance) {
         printf("Solution converged after %d iterations.\n", final_iteration);
     } else {
         printf("Reached maximum number of iterations. Solution did NOT converge.\n");
         printf("Note: This is normal if you are using the default settings.\n");
     }
     
     gs_verbose_printf("\t****  Parallel Gauss-Seidel calculation completed ****\n");
 }
 