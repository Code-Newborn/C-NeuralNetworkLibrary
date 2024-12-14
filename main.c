/*
 * @Reference    : https://github.com/DiJuMx/NeuralNetworkLibrary/tree/master
 * @ReferenceAuthor:
 * @Year         :
 * @             :
 * @Descripttion :
 * @version      :
 * @Author       : ChenZhao
 * @Date         : 2024-12-14 15:49:17
 * @LastEditors  : ChenZhao
 * @LastEditTime : 2024-12-14 15:57:48
 */
#include <stdio.h>
#include <stdlib.h>
#include "neuralNetwork.h"
#include <time.h>

void printArray( int arr[], int size ) {
    for ( int i = 0; i < size; i++ ) {
        printf( "%d ", arr[ i ] );
    }
    printf( "\n" );
}

int main() {
    /* code */
    printf( "Hello Neural Networks!\n" );

    int num_perlayer[ 3 ] = { 3, 2, 2 };  // Hidden层1、Hidden层2、输出层

    mlpNetwork* nn = createNetwork( 3, num_perlayer, 3, BPROP_LEARNING, LIN_ACTIVATION );

    double weights[ 26 ] = { 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,  // 全连接1 权重（含bias）
                             0, 1, 1, 1, 0, 1, 1, 1,              // 全连接2 权重（含bias）
                             0, 1, 1, 0, 1, 1 };                  // 全连接3 权重（含bias）
    setWeights( nn, weights );


    dataMember* dm;
    double      a[ 3 ] = { 10, 20, 30 };
    double      b[ 2 ] = { 0, 0 };
    double      c[ 2 ] = { 0, 0 };
    double      d[ 2 ] = { 0, 0 };
    dm->inputs         = a;
    dm->targets        = b;
    dm->outputs        = c;
    dm->errors         = d;

    clock_t start, end;
    double  cpu_time_used;

    start = clock();

    computeNetwork( nn, dm, 3, 2 );// 正向传播一次

    end = clock();

    cpu_time_used = ( double )( end - start );
    printf( "运行时间：%.3f 毫秒\n", cpu_time_used );

    printf( "%.3f\n", nn->layers[ 0 ][ 0 ].weights[ 0 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 0 ].weights[ 1 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 0 ].weights[ 2 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 0 ].weights[ 3 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 1 ].weights[ 0 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 1 ].weights[ 1 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 1 ].weights[ 2 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 1 ].weights[ 3 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 2 ].weights[ 0 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 2 ].weights[ 1 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 2 ].weights[ 2 ] );
    printf( "%.3f\n", nn->layers[ 0 ][ 2 ].weights[ 3 ] );
    printf( "\n" );

    printf( "%.3f\n", nn->layers[ 1 ][ 0 ].weights[ 0 ] );
    printf( "%.3f\n", nn->layers[ 1 ][ 0 ].weights[ 1 ] );
    printf( "%.3f\n", nn->layers[ 1 ][ 0 ].weights[ 2 ] );

    printf( "%.3f\n", nn->layers[ 1 ][ 1 ].weights[ 0 ] );
    printf( "%.3f\n", nn->layers[ 1 ][ 1 ].weights[ 1 ] );
    printf( "%.3f\n", nn->layers[ 1 ][ 1 ].weights[ 2 ] );
    printf( "\n" );

    printf( "%.3f\n", nn->layers[ 2 ][ 0 ].weights[ 0 ] );
    printf( "%.3f\n", nn->layers[ 2 ][ 0 ].weights[ 1 ] );
    printf( "%.3f\n", nn->layers[ 2 ][ 0 ].weights[ 2 ] );

    printf( "%.3f\n", nn->layers[ 2 ][ 1 ].weights[ 0 ] );
    printf( "%.3f\n", nn->layers[ 2 ][ 1 ].weights[ 1 ] );
    printf( "%.3f\n", nn->layers[ 2 ][ 1 ].weights[ 2 ] );
    printf( "\n" );

    printf( "%.3f\n", nn->layers[ 0 ][ 0 ].output );
    printf( "%.3f\n", nn->layers[ 0 ][ 1 ].output );
    printf( "%.3f\n", nn->layers[ 0 ][ 2 ].output );
    printf( "\n" );

    printf( "%.3f\n", nn->layers[ 1 ][ 0 ].output );
    printf( "%.3f\n", nn->layers[ 1 ][ 1 ].output );
    printf( "\n" );

    printf( "%.3f\n", nn->layers[ 2 ][ 0 ].output );
    printf( "%.3f\n", nn->layers[ 2 ][ 1 ].output );


    destroyNet( nn );


    system( "pause" );  // 按任意键退出控制台
    return 0;
}
