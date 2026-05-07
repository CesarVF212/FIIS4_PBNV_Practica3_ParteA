#include "mathSSE.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

// Autor: Caesar

//tomar tiempos de inicio
#define TIME_INIT(TimeSuffix) \
    auto start_##TimeSuffix = std::chrono::high_resolution_clock::now();
//tomar tiempos de final y calcular milisegundos
#define TIME_END(TimeSuffix) \
    auto end_##TimeSuffix = std::chrono::high_resolution_clock::now();\
    std::chrono::duration<double> duration_##TimeSuffix = end_##TimeSuffix - start_##TimeSuffix;\
    double milliseconds_##TimeSuffix = duration_##TimeSuffix / 1ms;
#define PRETTY_PRINT_TIME(TimeSuffix, message) \
    std::cout << "Execution time "<<message<<" :" << milliseconds_##TimeSuffix << " milliseconds." << std::endl;

using namespace std::chrono_literals;

#define NUM_MATRIX 1000
#define NUM_MATRIX_GPU 1000000

#define THREADS_PER_BLOCK 256

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

// Kernel GPU: cada thread calcula una multiplicación A1[i] * A2[i].
__global__ void multiplicaMatricesGPU_kernel(matrix4x4f* A1_d, matrix4x4f* A2_d, matrix4x4f* AResultadosGPU_d, unsigned int arraySize)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < arraySize)
    {
        AResultadosGPU_d[i] = multMatrix_tradicional(A1_d[i], A2_d[i]);
    }
}

// Funcion host: reserva memoria de GPU, copia datos, lanza el kernel y recupera resultados.
void multiplicaMatricesGPU(matrix4x4f* A1, matrix4x4f* A2, matrix4x4f* AResultadosGPU_h, unsigned int arraySize)
{
    matrix4x4f* A1_d;
    matrix4x4f* A2_d;
    matrix4x4f* AResultadosGPU_d;

    size_t bytes = arraySize * sizeof(matrix4x4f);

    // Reservamos memoria en la GPU.
    cudaMalloc((void**)&A1_d, bytes);
    cudaMalloc((void**)&A2_d, bytes);
    cudaMalloc((void**)&AResultadosGPU_d, bytes);

    // Copiamos los datos de entrada a la GPU.
    cudaMemcpy(A1_d, A1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(A2_d, A2, bytes, cudaMemcpyHostToDevice);

    // Calculamos numero de bloques.
    int threadsPerBlock = THREADS_PER_BLOCK;
    int numBlocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    // Lanzamos el kernel.
    multiplicaMatricesGPU_kernel<<<numBlocks, threadsPerBlock>>>(A1_d, A2_d, AResultadosGPU_d, arraySize);

    // Esperamos a que termine la GPU.
    cudaDeviceSynchronize();

    // Copiamos resultados de vuelta al host.
    cudaMemcpy(AResultadosGPU_h, AResultadosGPU_d, bytes, cudaMemcpyDeviceToHost);

    // Liberamos memoria de GPU.
    cudaFree(A1_d);
    cudaFree(A2_d);
    cudaFree(AResultadosGPU_d);
}

int main(int args, char** argv)
{
    // --- PARTE CPU --- //

    // Creamos 1000 matrices.
    matrix4x4f* A1 = new matrix4x4f[NUM_MATRIX];
    matrix4x4f* A2 = new matrix4x4f[NUM_MATRIX];
    matrix4x4f* AResultados = new matrix4x4f[NUM_MATRIX];

    // A las A las rellenamos con valores aleatorios. Vamos a permitir floats entre -100 y 100.
    A1 = fillRandom(A1, NUM_MATRIX);
    A2 = fillRandom(A2, NUM_MATRIX);

    TIME_INIT(MultiplicarMatricesCPU);
   
    // Llamamos de forma iterada a la multiplicacion de matrices tradicional.
    for(int iter = 0; iter < NUM_MATRIX; iter++)
    {
        AResultados[iter] = multMatrix_tradicional(A1[iter], A2[iter]);
    }
    TIME_END(MultiplicarMatricesCPU);
    PRETTY_PRINT_TIME(MultiplicarMatricesCPU,"Tiempo en CPU");


    // --- PARTE 1 - IMPLEMENTACION GPU --- //

    // Creamos 1000000 matrices para la version GPU.
    matrix4x4f* A1_GPU = new matrix4x4f[NUM_MATRIX_GPU];
    matrix4x4f* A2_GPU = new matrix4x4f[NUM_MATRIX_GPU];
    matrix4x4f* AResultadosGPU_h = new matrix4x4f[NUM_MATRIX_GPU];

    A1_GPU = fillRandom(A1_GPU, NUM_MATRIX_GPU);
    A2_GPU = fillRandom(A2_GPU, NUM_MATRIX_GPU);

    TIME_INIT(MultiplicarMatricesGPU);

    multiplicaMatricesGPU(A1_GPU, A2_GPU, AResultadosGPU_h, NUM_MATRIX_GPU);

    TIME_END(MultiplicarMatricesGPU);
    PRETTY_PRINT_TIME(MultiplicarMatricesGPU,"Tiempo en GPU");

    // --- SPEEDUP --- //
    
    // Para comparar de forma justa, escalamos el tiempo de CPU al mismo numero de matrices.
    double cpuEscalado = milliseconds_MultiplicarMatricesCPU * (double)(NUM_MATRIX_GPU / NUM_MATRIX);
    double speedup = cpuEscalado / milliseconds_MultiplicarMatricesGPU;
    std::cout << "Tiempo CPU extrapolado a " << NUM_MATRIX_GPU << " matrices: " << cpuEscalado << " ms" << std::endl;
    std::cout << "Speedup (CPU/GPU): " << speedup << std::endl;

    // Liberamos memoria.
    delete[] A1;
    delete[] A2;
    delete[] AResultados;
    delete[] A1_GPU;
    delete[] A2_GPU;
    delete[] AResultadosGPU_h;

    return 0;
}