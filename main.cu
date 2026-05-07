#include "mathSSE.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_MATRIX 1000

matrix4x4f* fillRandom(matrix4x4f* matrix, unsigned int arraySize)
{
    srand(time(NULL));

    for(unsigned int iter = 0; iter < arraySize; iter++)
    {
        for(unsigned int i = 0; i < matrix[0].size; i++)
        {
            for(unsigned int j = 0; j < matrix[0].size; j++)
            {
                float r = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
                matrix[iter].m_grid[i][j] = r;
            }
        }
    }

    return matrix;
}

int main(int args, char** argv)
{
    // Creamos 1000 matrices.
    matrix4x4f* A1 = new matrix4x4f[NUM_MATRIX];
    matrix4x4f* A2 = new matrix4x4f[NUM_MATRIX];
    matrix4x4f* AResultados = new matrix4x4f[NUM_MATRIX];

    // A las A las rellenamos con valores aleatorios. Vamos a permitir floats entre -100 y 100.
    A1 = fillRandom(A1, NUM_MATRIX);
    A2 = fillRandom(A2, NUM_MATRIX);

    // Llamamos de forma iterada a la multiplicacion de matrices tradicional.
    for(int iter = 0; iter < NUM_MATRIX; iter++)
    {
        AResultados[iter] = multMatrix_tradicional(A1[iter], A2[iter]);
    }

    printMatrix(AResultados[0]);
    
    return 0;
}