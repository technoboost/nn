#include<stdio.h>
#include<time.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
float input[4][3]={{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
float y[4]={0,1,1,0};
float weights1[3][4],weights2[4],output[4],layer1[4][4],bias1[4][4],bias2[4],alpha=10;
float sigmoid(float x)
{
     float exp_value;
     float return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}
 
float dsigmoid(float x)
{
    float return_value;
    return_value =x*(1-x);
    return return_value;
}


void main()
{
    int i,j,k,epoch;
    float product1=0,d_weights1[3][4],d_weights2[4],interm1[4],interm2[4][4];
    srand(time(0));
    for (i=0;i<3;i++)
    {
        for(j=0;j<4;j++)
        {
            weights1[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for(j=0;j<4;j++)
    {
            weights2[j]=(double)rand() / (double)RAND_MAX ;
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            bias1[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for(j=0;j<4;j++)
    {
            bias2[j]=(double)rand() / (double)RAND_MAX ;
    }
    for(epoch=0;epoch<100;epoch++)
    {
        
        /***********************FEEDFORWARD**************************/
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                for(k=0;k<3;k++)
                {
                    product1 = product1 + input[i][k]*weights1[k][j];
                }
                layer1[i][j]=sigmoid(product1+bias1[i][j]);
                product1=0;
            }
        }
        printf("\nOutput\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                product1 = product1 + layer1[i][j]*weights2[j];
            }
            output[i]=sigmoid(product1+bias2[i]);
            product1=0;
            printf("%f\t",output[i]);
        }
        printf("\n");
        
        
        /***********************BACKPROPAGATION****************************/
        for(i=0;i<4;i++)
        {
            interm1[i] = 2*(y[i] - output[i])*dsigmoid(output[i]);
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                product1 = product1 + layer1[j][i]*interm1[j];
            }
            d_weights2[i]=product1;
            product1=0;
            //printf("%f\t",d_weights2[i]);
        }
        printf("\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                product1 = interm1[i]*weights2[j];
                interm2[i][j]=product1*dsigmoid(layer1[i][j]);
                product1=0;
                //printf("%f\t",interm2[i][j]);
            }
            //printf("\n");
        }
        //printf("\n");
        //printf("\n");
        for (i=0;i<3;i++)
        {
            for(j=0;j<4;j++)
            {
                for(k=0;k<4;k++)
                {
                    product1 = product1 + input[k][i]*interm2[k][j];
                }
                d_weights1[i][j]=product1;
                product1=0;
            }
        }
        
        
        /*********************UPDATE WEIGHTS*****************************/
        for (i=0;i<3;i++)
        {
            for(j=0;j<4;j++)
            {
                weights1[i][j] +=alpha*d_weights1[i][j];
            }
        }
        for(j=0;j<4;j++)
        {
                weights2[j] +=alpha*d_weights2[j];
        }
        /*********************UPDATE BIAS*****************************/
                printf("\nBias1\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
               bias1[i][j] +=alpha*interm2[i][j];
               printf("%f\t",bias1[i][j]);
            }
            printf("\n");
        }
        printf("\nBias2\n");
        for(j=0;j<4;j++)
        {
                bias2[j] +=alpha*interm1[j];
                printf("%f\t",bias2[j]);
        }
        printf("\n");
    }
}
