
/*
    To run the code do this command:
    
    $ mpicc -fopenmp -o main main.c; mpiexec --use-hwthread-cpus -n 4 main; rm main
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MIN_OF_3(X, Y, Z) (MIN((X), MIN((Y), (Z))))

#define NUMBER_OF_COMM 2
#define WORD_LENGTH 10
#define THREAD_COUNT 4
#define N 1000000
#define Q 10

char database[N][WORD_LENGTH];
char queries[Q][WORD_LENGTH];

typedef struct
{
    int operations;
    char database_word[WORD_LENGTH];
} query_result;

query_result operations[Q][N];

int get_database(const int);
int get_queries(const int);

void distribute_database_between_communicators(const int, int *);
void distribute_queries_between_communicators(const int, int *);

int compare_function(const void *, const void *);
int levenshtein_distance(const char *, const char *);
void calculate_levenshtein_distance(const int, const int);

void gather_query_results(const int, int *, int *);
void print_query_results(const int, const int, const int);

int main(int argc, char *argv[])
{
	freopen("input.txt", "r", stdin);
	
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;

    // Get the rank and size in the original communicator.
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Determine color based on row.
    int color = world_rank / NUMBER_OF_COMM;

    MPI_Comm row_comm;

    // Split the communicator based on the color and use the original rank for ordering.
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &row_comm);

    int row_rank;
    int row_size;

    // Get the rank and size in the row communicator.
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    int database_size = get_database(world_rank);
    int queries_size = get_queries(world_rank);

    distribute_database_between_communicators(world_rank, &database_size);
    distribute_queries_between_communicators(world_rank, &queries_size);

    // Distribute database between nodes in the communicator.
    database_size /= NUMBER_OF_COMM;

    MPI_Bcast(&database_size, 1, MPI_INT, 0, row_comm);
    MPI_Scatter(database, database_size * WORD_LENGTH, MPI_CHAR,
                database, database_size * WORD_LENGTH, MPI_CHAR,
                0, row_comm);

    // Distribute queries between nodes in the communicator.
    MPI_Bcast(&queries_size, 1, MPI_INT, 0, row_comm);
    MPI_Bcast(queries, queries_size * WORD_LENGTH, MPI_CHAR, 0, row_comm);
    
    // printf("From node %i, n = %i\n", world_rank, database_size);
	// printf("From node %i, q = %i\n", world_rank, queries_size);

	// for (int i = 0; i < database_size; ++i)
	// {
	// 	printf("From node %i, database[%i] = %s\n", world_rank, i, database[i]);
	// }

	// for (int j = 0; j < queries_size; ++j)
	// {
	// 	printf("From node %i, queries[%i] = %s\n", world_rank, j, queries[j]);
	// }

    calculate_levenshtein_distance(queries_size, database_size);

    // Gather queries first MIN(database_size, 10) of the operations on node 0 from other nodes in the communicator.
    for (int j = 0; j < queries_size; ++j)
    {
        int top_ten_count = MIN(database_size, 10);

        MPI_Gather(&operations[j], top_ten_count * sizeof(query_result), MPI_CHAR,
                   &operations[j], top_ten_count * sizeof(query_result), MPI_CHAR,
                   0, row_comm);

        if (row_rank == 0)
        {
            qsort(operations[j], top_ten_count * row_size, sizeof(query_result), compare_function);
        }
    }

    // Gather queries on node 0 from other nodes in the original communicator.
    gather_query_results(world_rank, &queries_size, &database_size);

    print_query_results(world_rank, queries_size, database_size);

    MPI_Comm_free(&row_comm);
    MPI_Finalize();

    return EXIT_SUCCESS;
}

int get_database(const int world_rank)
{
    int database_size;

    if (world_rank == 0)
    {
        scanf("%i", &database_size);

        for (int i = 0; i < database_size; ++i)
        {
            scanf("%s", database[i]);
        }
    }

    return database_size;
}

int get_queries(const int world_rank)
{
    int queries_size;

    if (world_rank == 0)
    {
        scanf("%i", &queries_size);

        for (int i = 0; i < queries_size; ++i)
        {
            scanf("%s", queries[i]);
        }
    }

    return queries_size;
}

void distribute_database_between_communicators(const int world_rank, int *database_size)
{
    if (world_rank == 0)
    {
        MPI_Send(database_size, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        MPI_Send(database, *database_size * WORD_LENGTH, MPI_CHAR, 2, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 2)
    {
        MPI_Recv(database_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Recv(database, *database_size * WORD_LENGTH, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    }
}

void distribute_queries_between_communicators(const int world_rank, int *queries_size)
{
    if (world_rank == 0)
    {
        *queries_size /= NUMBER_OF_COMM;

        MPI_Send(queries_size, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        MPI_Send(&queries[*queries_size], *queries_size * WORD_LENGTH, MPI_CHAR, 2, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 2)
    {
        MPI_Recv(queries_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Recv(queries, *queries_size * WORD_LENGTH, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    }
}

int compare_function(const void *x, const void *y)
{
    return ((query_result *)x)->operations - ((query_result *)y)->operations;
}

int levenshtein_distance(const char *word1, const char *word2)
{
    int word1_length = strlen(word1);
    int word2_length = strlen(word2);

    int **dp_talbe = malloc((word2_length + 1) * sizeof(int *));

#pragma omp parallel for num_threads(THREAD_COUNT)
    for (int i = 0; i <= word2_length; ++i)
    {
        dp_talbe[i] = malloc((word1_length + 1) * sizeof(int));
        dp_talbe[i][0] = i;
    }

#pragma omp parallel for num_threads(THREAD_COUNT)
    for (int j = 1; j <= word1_length; ++j)
    {
        dp_talbe[0][j] = j;
    }

#pragma omp parallel for num_threads(THREAD_COUNT)
    for (int i = 1; i <= word2_length; ++i)
    {
#pragma omp parallel for num_threads(THREAD_COUNT)
        for (int j = 1; j <= word1_length; ++j)
        {
            dp_talbe[i][j] = MIN_OF_3(dp_talbe[i - 1][j] + 1,
                                dp_talbe[i][j - 1] + 1,
                                dp_talbe[i - 1][j - 1] + ((word1[j - 1] == word2[i - 1]) ? 0 : 1));
        }
    }

    return dp_talbe[word2_length][word1_length];
}

void calculate_levenshtein_distance(const int queries_size, const int database_size)
{
#pragma omp parallel for num_threads(THREAD_COUNT)
    for (int j = 0; j < queries_size; ++j)
    {
#pragma omp parallel for num_threads(THREAD_COUNT)
        for (int i = 0; i < database_size; ++i)
        {
            operations[j][i].operations = levenshtein_distance(database[i], queries[j]);

            strcpy(operations[j][i].database_word, database[i]);
        }

        qsort(operations[j], database_size, sizeof(query_result), compare_function);
    }
}

void gather_query_results(const int world_rank, int *queries_size, int *database_size)
{
    if (world_rank == 0)
    {
        MPI_Recv(&operations[*queries_size], *queries_size * N * sizeof(query_result), MPI_CHAR,
                 2, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        *queries_size *= NUMBER_OF_COMM;
        *database_size *= NUMBER_OF_COMM;
    }
    else if (world_rank == 2)
    {
        MPI_Send(operations, *queries_size * N * sizeof(query_result), MPI_CHAR,
                 0, 0, MPI_COMM_WORLD);
    }
}

void print_query_results(const int world_rank, const int queries_size, const int database_size)
{
    if (world_rank == 0)
    {
        for (int j = 0; j < queries_size; ++j)
        {
            printf("Top %i results to query[%i] = %s =>\n", MIN(database_size, 10), j, queries[j]);

            for (int i = 0; i < MIN(database_size, 10); ++i)
            {
                printf("%s operations = %i\n", operations[j][i].database_word, operations[j][i].operations);
            }

            printf("==================================================\n");
        }
    }
}
