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
double beta_1 = 0.9;
double beta_2 = 0.999;
double epsilon = 1e-8;
double weights1_m_t[3][4];
double weights2_m_t[4][5];
double weights3_m_t[5];
double weights1_v_t[3][4];
double weights2_v_t[4][5];
double weights3_v_t[5];
double bias1_m_t[4][4];
double bias2_m_t[4][5];
double bias3_m_t[5];
double bias1_v_t[4][4];
double bias2_v_t[4][5];
double bias3_v_t[5];
int t = 0;
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
    double weights_prev;
    double bias_prev;
    double m_cap,v_cap;
    srand(time(0));
    for (i=0;i<3;i++)
    {
        for(j=0;j<4;j++)
        {
            weights1[i][j]=(double)rand() / (double)RAND_MAX ;
            weights1_m_t[i][j]=0;
            weights1_v_t[i][j]=0;
        }
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<5;j++)
        {
            weights2[i][j]=(double)rand() / (double)RAND_MAX ;
            weights2_m_t[i][j]=0;
            weights2_v_t[i][j]=0;
        }
    }
    for(j=0;j<4;j++)
    {
            weights3[j]=(double)rand() / (double)RAND_MAX ;
            weights3_m_t[j]=0;
            weights3_v_t[j]=0;
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            bias1[i][j]=0 ;
            bias1_m_t[i][j]=0;
            bias1_v_t[i][j]=0;
        }
    }
    for (i=0;i<5;i++)
    {
        for(j=0;j<5;j++)
        {
            bias2[i][j]=0 ;
            bias2_m_t[i][j]=0;
            bias2_v_t[i][j]=0;
        }
    }
    for(j=0;j<4;j++)
    {
            bias3[j]=0 ;
            bias3_m_t[j]=0;
            bias3_v_t[j]=0;
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
        t+=1;
        for (i=0;i<3;i++)
        {
            for(j=0;j<4;j++)
            {
                weights1_m_t[i][j] = beta_1*weights1_m_t[i][j] + (1-beta_1)*d_weights1[i][j];	//updates the moving averages of the gradient
	            weights1_v_t[i][j] = beta_2*weights1_v_t[i][j] + (1-beta_2)*(d_weights1[i][j]*d_weights1[i][j]);	//updates the moving averages of the squared gradient
	            m_cap = weights1_m_t[i][j]/(1-(pow(beta_1,t)));		//calculates the bias-corrected estimates
	            v_cap = weights1_v_t[i][j]/(1-(pow(beta_2,t)));		//calculates the bias-corrected estimates
	            weights_prev = weights1[i][j];
	            weights1[i][j] = weights1[i][j] + (alpha*m_cap)/(sqrt(v_cap)+epsilon);	//updates the parameters
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                weights2_m_t[i][j] = beta_1*weights2_m_t[i][j] + (1-beta_1)*d_weights2[i][j];	//updates the moving averages of the gradient
	            weights2_v_t[i][j] = beta_2*weights2_v_t[i][j] + (1-beta_2)*(d_weights2[i][j]*d_weights2[i][j]);	//updates the moving averages of the squared gradient
	            m_cap = weights2_m_t[i][j]/(1-(pow(beta_1,t)));		//calculates the bias-corrected estimates
	            v_cap = weights2_v_t[i][j]/(1-(pow(beta_2,t)));		//calculates the bias-corrected estimates
	            weights_prev = weights2[i][j];
	            weights2[i][j] = weights2[i][j] + (alpha*m_cap)/(sqrt(v_cap)+epsilon);	//updates the parameters
            }
        }
        for(j=0;j<4;j++)
        {
                weights3_m_t[j] = beta_1*weights3_m_t[j] + (1-beta_1)*d_weights3[j];	//updates the moving averages of the gradient
	            weights3_v_t[j] = beta_2*weights3_v_t[j] + (1-beta_2)*(d_weights3[j]*d_weights3[j]);	//updates the moving averages of the squared gradient
	            m_cap = weights3_m_t[j]/(1-(pow(beta_1,t)));		//calculates the bias-corrected estimates
	            v_cap = weights3_v_t[j]/(1-(pow(beta_2,t)));		//calculates the bias-corrected estimates
	            weights_prev = weights3[j];
	            weights3[j] = weights3[j] + (alpha*m_cap)/(sqrt(v_cap)+epsilon);	//updates the parameters
        }
        /*********************UPDATE BIAS*****************************/
          //      printf("\nBias1\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                bias1_m_t[i][j] = beta_1*bias1_m_t[i][j] + (1-beta_1)*interm3[i][j];	//updates the moving averages of the gradient
	            bias1_v_t[i][j] = beta_2*bias1_v_t[i][j] + (1-beta_2)*(interm3[i][j]*interm3[i][j]);	//updates the moving averages of the squared gradient
	            m_cap = bias1_m_t[i][j]/(1-(pow(beta_1,t)));		//calculates the bias-corrected estimates
	            v_cap = bias1_v_t[i][j]/(1-(pow(beta_2,t)));		//calculates the bias-corrected estimates
	            bias_prev = bias1[i][j];
	            bias1[i][j] = bias1[i][j] + (alpha*m_cap)/(sqrt(v_cap)+epsilon);	//updates the parameters
            }
            //printf("\n");
        }
        //printf("\nBias2\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                bias2_m_t[i][j] = beta_1*bias2_m_t[i][j] + (1-beta_1)*interm2[i][j];	//updates the moving averages of the gradient
	            bias2_v_t[i][j] = beta_2*bias2_v_t[i][j] + (1-beta_2)*(interm2[i][j]*interm2[i][j]);	//updates the moving averages of the squared gradient
	            m_cap = bias2_m_t[i][j]/(1-(pow(beta_1,t)));		//calculates the bias-corrected estimates
	            v_cap = bias2_v_t[i][j]/(1-(pow(beta_2,t)));		//calculates the bias-corrected estimates
	            bias_prev = bias2[i][j];
	            bias2[i][j] = bias2[i][j] + (alpha*m_cap)/(sqrt(v_cap)+epsilon);	//updates the parameters
            }
            //printf("\n");
        }
        //printf("\nBias3\n");
        for(j=0;j<4;j++)
        {
                bias3_m_t[j] = beta_1*bias3_m_t[j] + (1-beta_1)*interm1[j];	//updates the moving averages of the gradient
	            bias3_v_t[j] = beta_2*bias3_v_t[j] + (1-beta_2)*(interm1[j]*interm1[j]);	//updates the moving averages of the squared gradient
	            m_cap = bias3_m_t[j]/(1-(pow(beta_1,t)));		//calculates the bias-corrected estimates
	            v_cap = bias3_v_t[j]/(1-(pow(beta_2,t)));		//calculates the bias-corrected estimates
	            bias_prev = bias3[j];
	            bias3[j] = bias3[j] + (alpha*m_cap)/(sqrt(v_cap)+epsilon);	//updates the parameters
        }
        printf("\n");
    }
    
    /***********TESTING********************/
    double input1[4][3]={{0,1,1},{0,0,1},{1,1,1},{1,0,1}};/* 1 0 0 1*/
    for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                for(k=0;k<3;k++)
                {
                    product1 = product1 + input1[i][k]*weights1[k][j];
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
        
}
