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
void main()
{
    double **weights1,**mat,**input,product1,layer1[4][4];
    int i,j,k;
    srand(time(0));
    weights1 = (double **)malloc(3 * sizeof(double *)); 
        for (i=0; i<3; i++) 
        {
             weights1[i] = (double *)malloc(4 * sizeof(double)); 
        }
    input = (double **)malloc(4 * sizeof(double *)); 
        for (i=0; i<4; i++) 
        {
             input[i] = (double *)malloc(3 * sizeof(double)); 
        }
     mat = (double **)malloc(4 * sizeof(double *)); 
        for (i=0; i<4; i++) 
        {
             mat[i] = (double *)malloc(4 * sizeof(double)); 
        }
    for (i=0;i<4;i++)
    {
        for(j=0;j<3;j++)
        {
            *(*(input+i)+j)=x[i][j];
        }
    }
    for (i=0;i<3;i++)
    {
        for(j=0;j<4;j++)
        {
            //weights1[i][j]=
            weights1[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    matmul(4,3,3,4,mat,input,weights1);
    printf("\nThe value of matrix 'C' = \n");

        for(i=0;i<4;i++)
        {
         printf("\n");
         for(j=0;j<4;j++)
         printf("%lf\t",*(*(mat+i)+j));
         }
         
         
        /****Freeing double pointer**********/
        for (int i=0; i<4; ++i) {
        free(mat[i]);
        }
        free(mat);
        
        
        
        
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
}
