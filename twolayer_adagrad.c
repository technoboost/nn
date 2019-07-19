#include<stdio.h>
#include<time.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
double input[4][3]={{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
double y[4]={0,1,1,0};
double weights1[3][4];
double weights2[4][5];
double weights3[5];
double output[4];
double layer1[4][4];
double layer2[4][5];
double bias1[4][4];
double bias2[4][5];
double bias3[5];
double alpha=0.1;
double weights1_v[3][4];
double weights2_v[4][5];
double weights3_v[5];
double bias1_v[4][4];
double bias2_v[4][5];
double bias3_v[5];
double weights1_ad[3][4];
double weights2_ad[4][5];
double weights3_ad[5];
double bias1_ad[4][4];
double bias2_ad[4][5];
double bias3_ad[5];
double epsilone=1e-8;
double LeakyRelu(double x)
{
    if (x >= 0)
        return x;
    else
        return x / 20;
}
double dLeakyRelu(double x)
{
    if (x >= 0)
        return 1;
    else
        return 1.0 / 20;
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


void main()
{
    int i,j,k,epoch;
    double product1=0;
    double d_weights1[3][4];
    double d_weights2[4][5];
    double d_weights3[5];
    double interm1[4];
    double interm2[4][5];
    double interm3[4][4];
    srand(time(0));
    for (i=0;i<3;i++)
    {
        for(j=0;j<4;j++)
        {
            weights1[i][j]=(double)rand() / (double)RAND_MAX ;
            weights1_v[i][j]=0;
            weights1_ad[i][j]=0;
        }
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<5;j++)
        {
            weights2[i][j]=(double)rand() / (double)RAND_MAX ;
            weights2_v[i][j]=0;
            weights2_ad[i][j]=0;
        }
    }
    for(j=0;j<4;j++)
    {
            weights3[j]=(double)rand() / (double)RAND_MAX ;
            weights3_v[j]=0;
            weights3_ad[j]=0;
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            bias1[i][j]=(double)rand() / (double)RAND_MAX ;
            bias1_v[i][j]=0;
            bias1_ad[i][j]=0;
        }
    }
    for (i=0;i<5;i++)
    {
        for(j=0;j<5;j++)
        {
            bias2[i][j]=(double)rand() / (double)RAND_MAX ;
            bias2[i][j]=0;
            bias2_ad[i][j]=0;
        }
    }
    for(j=0;j<4;j++)
    {
            bias3[j]=(double)rand() / (double)RAND_MAX ;
            bias3_v[j]=0;
            bias3_ad[j]=0;
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
                weights1_ad[i][j]=weights1_ad[i][j] + d_weights1[i][j]*d_weights1[i][j];
                weights1_v[i][j]=alpha*d_weights1[i][j]/(sqrt(weights1_ad[i][j]+epsilone));
                weights1[i][j] +=weights1_v[i][j];
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                weights2_ad[i][j]=weights2_ad[i][j] + d_weights2[i][j]*d_weights2[i][j];
                weights2_v[i][j]=alpha*d_weights2[i][j]/(sqrt(weights2_ad[i][j]+epsilone));
                weights2[i][j] +=weights2_v[i][j];
            }
        }
        for(j=0;j<4;j++)
        {
                weights3_ad[j]=weights3_ad[j] + d_weights3[j]*d_weights3[j];
                weights3_v[j]=alpha*d_weights3[j]/(sqrt(weights3_ad[j]+epsilone));
                weights3[j] +=weights3_v[j];
        }
        /*********************UPDATE BIAS*****************************/
          //      printf("\nBias1\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                 bias1_ad[i][j]=bias1_ad[i][j] + interm3[i][j]*interm3[i][j];
                 bias1_v[i][j]=alpha*interm3[i][j]/(sqrt(bias1_ad[i][j]+epsilone));
                 bias1[i][j] +=bias1_v[i][j];
            }
            //printf("\n");
        }
        //printf("\nBias2\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                 bias2_ad[i][j]=bias2_ad[i][j] + interm2[i][j]*interm2[i][j];
                 bias2_v[i][j]=alpha*interm2[i][j]/(sqrt(bias2_ad[i][j]+epsilone));
                 bias2[i][j] +=bias2_v[i][j];
            }
            //printf("\n");
        }
        //printf("\nBias3\n");
        for(j=0;j<4;j++)
        {
                 bias3_ad[j]=bias3_ad[j] + interm1[j]*interm1[j];
                 bias3_v[j]=alpha*interm1[j]/(sqrt(bias3_ad[j]+epsilone));
                 bias3[j] +=bias3_v[j];
        }
        printf("\n");
    }
}
