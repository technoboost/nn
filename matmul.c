#include<stdio.h>
#include <stdlib.h>
#include <time.h>
double x[4][3]={{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
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
void doublemalloc(double ***var,int row,int col)
{
    int i;
    *var = (double **)malloc(row * sizeof(double *)); 
        for (i=0; i<row; i++) 
        {
             (*var)[i] = (double *)malloc(col * sizeof(double)); 
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

void main()
{
    double **weights1,**mat,**input,product1,layer1[4][4];
    int i,j,k;
    srand(time(0));
    doublemalloc(&weights1,3,4);
    doublemalloc(&input,4,3);
    doublemalloc(&mat,4,4);
    for (i=0;i<4;i++)
    {
        for(j=0;j<3;j++)
        {
            *(*(input+i)+j)=x[i][j];
        }
    }
    doublerandom(&weights1, 3, 4);
    matmul(4,3,3,4,mat,input,weights1);
    printf("\nThe value of matrix 'C' = \n");

        for(i=0;i<4;i++)
        {
         printf("\n");
         for(j=0;j<4;j++)
         printf("%lf\t",*(*(mat+i)+j));
         }
    
        printf("\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                for(k=0;k<3;k++)
                {
                    product1 = product1 + input[i][k]*weights1[k][j];
                }
                                layer1[i][j]=product1;
                product1=0;
            }
        }
     for(i=0;i<4;i++)
        {
         printf("\n");
         for(j=0;j<4;j++)
         printf("%lf\t",layer1[i][j]);
         }
    doublefree(&mat,4);
    doublefree(&input,4);
    doublefree(&weights1,3);
}
