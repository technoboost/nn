#include<stdio.h>
#include<time.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
float input[4][3]={{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
float y[4]={0,1,1,0};
float weights1[3][4];
float weights2[4][5];
float weights3[5];
float output[4];
float layer1[4][4];
float layer2[4][5];
float bias1[4][4];
float bias2[4][5];
float bias3[5];
float alpha=0.1;
float LeakyRelu(float x)
{
    if (x >= 0)
        return x;
    else
        return x / 20;
}
float dLeakyRelu(float x)
{
    if (x >= 0)
        return 1;
    else
        return 1.0 / 20;
}
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
    float product1=0;
    float d_weights1[3][4];
    float d_weights2[4][5];
    float d_weights3[5];
    float interm1[4];
    float interm2[4][5];
    float interm3[4][4];
    srand(time(0));
    for (i=0;i<3;i++)
    {
        for(j=0;j<4;j++)
        {
            weights1[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<5;j++)
        {
            weights2[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for(j=0;j<4;j++)
    {
            weights3[j]=(double)rand() / (double)RAND_MAX ;
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            bias1[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for (i=0;i<5;i++)
    {
        for(j=0;j<5;j++)
        {
            bias2[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for(j=0;j<4;j++)
    {
            bias3[j]=(double)rand() / (double)RAND_MAX ;
    }
    for(epoch=0;epoch<5000;epoch++)
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
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                for(k=0;k<4;k++)
                {
                    product1 = product1 + layer1[i][k]*weights2[k][j];
                }
                layer2[i][j]=LeakyRelu(product1+bias2[i][j]);
                product1=0;
            }
        }
        printf("\nOutput\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                product1 = product1 + layer2[i][j]*weights3[j];
            }
            output[i]=sigmoid(product1+bias3[i]);
            product1=0;
            printf("%f\t",output[i]);
        }
        printf("\n");
        
        
        /***********************BACKPROPAGATION****************************/
        for(i=0;i<4;i++)
        {
            interm1[i] = 2*(y[i] - output[i])*dsigmoid(output[i]);
        }
        for (i=0;i<5;i++)
        {
            for(j=0;j<4;j++)
            {
                product1 = product1 + layer2[j][i]*interm1[j];
            }
            d_weights3[i]=product1;
            product1=0;
        }
        printf("\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                product1 = interm1[i]*weights3[j];
                interm2[i][j]=product1*dLeakyRelu(layer2[i][j]);
                product1=0;
            }
        }
        //printf("\n");
        //printf("\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                for(k=0;k<4;k++)
                {
                    product1 = product1 + layer1[k][i]*interm2[k][j];
                }
                d_weights2[i][j]=product1;
                product1=0;
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                for(k=0;k<5;k++)
                {
                product1 = product1 + interm2[i][k]*weights2[j][k];
                }
                interm3[i][j]=product1*dsigmoid(layer1[i][j]);
                product1=0;
            }
        }
        //printf("\n");
        //printf("\n");
        for (i=0;i<3;i++)
        {
            for(j=0;j<4;j++)
            {
                for(k=0;k<4;k++)
                {
                    product1 = product1 + input[k][i]*interm3[k][j];
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
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                weights2[i][j] +=alpha*d_weights2[i][j];
            }
        }
        for(j=0;j<4;j++)
        {
                weights3[j] +=alpha*d_weights3[j];
        }
        /*********************UPDATE BIAS*****************************/
          //      printf("\nBias1\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
               bias1[i][j] +=alpha*interm3[i][j];
               //printf("%f\t",bias1[i][j]);
            }
            //printf("\n");
        }
        //printf("\nBias2\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
               bias2[i][j] +=alpha*interm2[i][j];
               //printf("%f\t",bias2[i][j]);
            }
            //printf("\n");
        }
        //printf("\nBias3\n");
        for(j=0;j<4;j++)
        {
                bias3[j] +=alpha*interm1[j];
                //printf("%f\t",bias3[j]);
        }
        printf("\n");
    }
}
