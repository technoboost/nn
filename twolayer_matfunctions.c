#include<stdio.h>
#include<time.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
double x[4][3]={{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
double y[4]={0,1,1,0};
double **input;
double **weights1;
double **weights2;
double **weights3;
double **output;
double **layer1;
double **layer2;
double **bias1;
double **bias2;
double **bias3;
double alpha=0.1;
void matmul(int rowA,int colA,int rowB,int colB, double *ans[], double *first[], double *second[])
{
        int i,j,k=0;
        if(colA!=rowB)
        {
            printf("Error => colA must be equal to rowB\n");
        }
        for(i=0;i<rowA;i++)
        {
            for(j=0;j<colB;j++)
            {
                *(*(ans+i)+j)=0;
                for(k=0;k<rowB;k++)
                {
                        *(*(ans+i)+j) = *(*(ans+i)+j) + (*(*(first+i) + k )) * (*(*(second+k) + j));
                }
            }//j
        }//i
}
void matsum(int row,int col, double *ans[], double *first[], double *second[])
{
    int i,j;
    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
             ans[i][j]=first[i][j]+second[i][j];
        }
    } 
}

void doublemalloc(double ***var,int row,int col)
{
    int i;
    *var = (double **)malloc(row * sizeof(double *)); 
        for (i=0; i<row; i++) 
        {
             (*var)[i] = (double *)malloc(col * sizeof(double)); 
        }
}

void mattranspose(int row, int col, double *ans[], double *matrix[])
{
    int i,j;
    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
             ans[j][i]=matrix[i][j];
        }
    }
}
void doublerandom(double ***var, int row, int col)
{
    int i,j;
    for (i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
            (*var)[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
}
void doublefree(double ***var, int row)
{
    int i;
    for (int i=0; i<row; i++) 
    {
        free((*var)[i]);
    }
    free(*var);
}

double sigmoid(double x)
{
     double exp_value;
     double return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}
 
double dsigmoid(double x)
{
    double return_value;
    return_value =x*(1-x);
    return return_value;
}
void doublematsigmoid(int row, int col,double *ans[], double *matrix[])
{
    int i,j;
    for(i=0; i<row; i++)
    {
        for(j=0; j<col; j++)
        {
            ans[i][j]=sigmoid(matrix[i][j]);
        }
    }
}
void doublematdsigmoid(int row, int col,double *ans[], double *matrix[])
{
    int i,j;
    for(i=0; i<row; i++)
    {
        for(j=0; j<col; j++)
        {
            ans[i][j]*=dsigmoid(matrix[i][j]);
        }
    }
}
void updateparams(int row, int col, double *ans[], double *params[])
{
    int i,j;
    for (i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
            ans[i][j] +=alpha*params[i][j];
        }
    }
}
void main()
{
    int i,j,k,epoch;
    double product1=0;
    double **d_weights1;
    double **d_weights2;
    double **d_weights3;
    double **interm1;
    double **interm2;
    double **interm3;
    double **tlayer1;
    double **tlayer2;
    double **tweights2;
    double **tweights3;
    double **tinput;
    srand(time(0));
    doublemalloc(&weights1,3,4);
    doublemalloc(&d_weights1,3,4);
    doublerandom(&weights1,3,4);
    doublemalloc(&weights2,4,5);
    doublemalloc(&d_weights2,4,5);
    doublerandom(&weights2,4,5);
    doublemalloc(&weights3,5,1);
    doublemalloc(&d_weights3,5,1);
    doublerandom(&weights3,5,1);
    doublemalloc(&bias1,4,4);
    doublerandom(&bias1,4,4);
    doublemalloc(&bias2,4,5);
    doublerandom(&bias2,4,5);
    doublemalloc(&bias3,4,1);
    doublerandom(&bias3,4,1);
    doublemalloc(&interm1,4,1);
    doublemalloc(&interm2,4,5);
    doublemalloc(&interm3,4,4);
    doublemalloc(&layer1,4,4);
    doublemalloc(&layer2,4,5);
    doublemalloc(&input,4,3);
    doublemalloc(&output,4,1);
    doublemalloc(&tlayer1,4,4);
    doublemalloc(&tlayer2,5,4);
    doublemalloc(&tweights2,5,4);
    doublemalloc(&tweights3,5,1);
    doublemalloc(&tinput,3,4);
    for (i=0;i<4;i++)
    {
        for(j=0;j<3;j++)
        {
            *(*(input+i)+j)=x[i][j];
        }
    }
    for(epoch=0;epoch<5000;epoch++)
    {
        
        /***********************FEEDFORWARD**************************/
        matmul(4,3,3,4,layer1,input,weights1);
        matsum(4,4,layer1,layer1,bias1);
        doublematsigmoid(4,4,layer1,layer1);
        matmul(4,4,4,5,layer2,layer1,weights2);
        matsum(4,5,layer2,layer2,bias2);
        doublematsigmoid(4,5,layer2,layer2);
        matmul(4,5,5,1,output,layer2,weights3);
        matsum(4,1,output,output,bias3);
        doublematsigmoid(4,1,output,output);
        printf("\nOutput\n");
        for (i=0;i<4;i++)
        {
            printf("%lf\t",output[i][0]);
        }
        printf("\n");
        
        
        /***********************BACKPROPAGATION****************************/
        for(i=0;i<4;i++)
        {
            interm1[i][0] = 2*(y[i] - output[i][0])*dsigmoid(output[i][0]);
        }
        mattranspose(4,5,tlayer2,layer2);
        matmul(5,4,4,1,d_weights3,tlayer2,interm1);
        mattranspose(5,1,tweights3,weights3);
        matmul(4,1,1,5,interm2,interm1,tweights3); 
        doublematdsigmoid(4,5,interm2,layer2);
        
        mattranspose(4,4,tlayer1,layer1);
        matmul(4,4,4,5,d_weights2,tlayer1,interm2);
        mattranspose(4,5,tweights2,weights2);
        matmul(4,5,5,4,interm3,interm2,tweights2);
        doublematdsigmoid(4,4,interm3,layer1);
        
        mattranspose(4,3,tinput,input);
        matmul(3,4,4,4,d_weights1,tinput,interm3);
        
        /*********************UPDATE WEIGHTS*****************************/
        updateparams(3,4,weights1,d_weights1);
        updateparams(4,5,weights2,d_weights2);
        updateparams(4,1,weights3,d_weights3);
        
        /*********************UPDATE BIAS*****************************/
        updateparams(4,4,bias1,interm3);
        updateparams(4,5,bias2,interm2);
        updateparams(4,1,bias3,interm1);
        printf("\n");
    }
}
