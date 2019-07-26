#include<stdio.h>
#include<time.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
double input[4][3]={{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
double y[4]={0,1,1,1};
double **input;
double **weights[7];
/*double weights1[3][4];
double weights2[4][5];
double weights3[5][6];
double weights4[6][6];
double weights5[6][6];
double weights6[6][4];
double weights7[4];*/
double **output;
/*double output[4];*/
double **layer[6];
/*double layer1[4][4];
double layer2[4][5];
double layer3[4][6];
double layer4[4][6];
double layer5[4][6];
double layer6[4][4];*/
double **bias[7];
/*double bias1[4][4];
double bias2[4][5];
double bias3[4][6];
double bias4[4][6];
double bias5[4][6];
double bias6[4][4];
double bias7[4];*/
double alpha=1;
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
    double **d_weights[7];
    /*double d_weights1[3][4];
    double d_weights2[4][5];
    double d_weights3[5][6];
    double d_weights4[6][6];
    double d_weights5[6][6];
    double d_weights6[6][4];
    double d_weights7[4];*/
    double **interm[7];
    /*double interm1[4];
    double interm2[4][4];
    double interm3[4][6];
    double interm4[4][6];
    double interm5[4][6];
    double interm6[4][5];
    double interm7[4][4];*/
    double **tlayer[6];
    double **tinput;
    
    srand(time(0));
    doublemalloc(&weights[0],3,4);
    doublemalloc(&d_weights[0],3,4);
    doublerandom(&weights[0],3,4);
    
    doublemalloc(&weights[1],4,5);
    doublemalloc(&d_weights[1],4,5);
    doublerandom(&weights[1],4,5);
    
    doublemalloc(&weights[2],5,6);
    doublemalloc(&d_weights[2],5,6);
    doublerandom(&weights[2],5,6);
    
    doublemalloc(&weights[3],6,6);
    doublemalloc(&d_weights[3],6,6);
    doublerandom(&weights[3],6,6);
    
    doublemalloc(&weights[4],6,6);
    doublemalloc(&d_weights[4],6,6);
    doublerandom(&weights[4],6,6);
    
    doublemalloc(&weights[5],6,4);
    doublemalloc(&d_weights[5],6,4);
    doublerandom(&weights[5],6,4);
    
    doublemalloc(&weights[6],4,1);
    doublemalloc(&d_weights[6],4,1);
    doublerandom(&weights[6],4,1);
    
    doublemalloc(&bias[0],4,4);
    doublerandom(&bias[0],4,4);
    doublemalloc(&bias[1],4,5);
    doublerandom(&bias[1],4,5);
    doublemalloc(&bias[2],4,6);
    doublerandom(&bias[2],4,6);
    doublemalloc(&bias[3],4,6);
    doublerandom(&bias[3],4,6);
    doublemalloc(&bias[4],4,6);
    doublerandom(&bias[4],4,6);
    doublemalloc(&bias[5],4,4);
    doublerandom(&bias[5],4,4);
    doublemalloc(&bias[6],4,1);
    doublerandom(&bias[6],4,1);

    doublemalloc(&interm[0],4,1);
    doublemalloc(&interm[1],4,4);
    doublemalloc(&interm[2],4,6); 
    doublemalloc(&interm[3],4,6);
    doublemalloc(&interm[4],4,6);
    doublemalloc(&interm[5],4,5);
    doublemalloc(&interm[6],4,4);
    
    doublemalloc(&layer[0],4,4);
    doublemalloc(&layer[1],4,5);
    doublemalloc(&layer[2],4,6);
    doublemalloc(&layer[3],4,6);
    doublemalloc(&layer[4],4,6);
    doublemalloc(&layer[5],4,4);
    doublemalloc(&input,4,3);
    doublemalloc(&output,4,1);
    doublemalloc(&tlayer[0],4,4);
    doublemalloc(&tlayer[1],5,4);
    doublemalloc(&tlayer[2],6,4);
    doublemalloc(&tlayer[3],6,4);
    doublemalloc(&tlayer[4],6,4);
    doublemalloc(&tlayer[5],4,4);
    doublemalloc(&tweights2,5,4);
    doublemalloc(&tinput,3,4);
    /*for (i=0;i<3;i++)
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
    for (i=0;i<5;i++)
    {
        for(j=0;j<6;j++)
        {
            weights3[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for (i=0;i<6;i++)
    {
        for(j=0;j<6;j++)
        {
            weights4[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for (i=0;i<6;i++)
    {
        for(j=0;j<6;j++)
        {
            weights5[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for (i=0;i<6;i++)
    {
        for(j=0;j<4;j++)
        {
            weights6[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for(j=0;j<4;j++)
    {
            weights7[j]=(double)rand() / (double)RAND_MAX ;
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            bias1[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<5;j++)
        {
            bias2[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<6;j++)
        {
            bias3[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<6;j++)
        {
            bias4[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<6;j++)
        {
            bias5[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for (i=0;i<4;i++)
    {
        for(j=0;j<4;j++)
        {
            bias6[i][j]=(double)rand() / (double)RAND_MAX ;
        }
    }
    for(j=0;j<4;j++)
    {
            bias7[j]=(double)rand() / (double)RAND_MAX ;
    }*/
    
    for(epoch=0;epoch<1000;epoch++)
    {
        
        /***********************FEEDFORWARD**************************/
        matmul(4,3,3,4,layer[0],input,weights[0]);
        matsum(4,4,layer[0],layer[0],bias[0]);
        doublematsigmoid(4,4,layer[0],layer[0]);
        matmul(4,4,4,5,layer[1],layer[0],weights[1]);
        matsum(4,5,layer[1],layer[1],bias[1]);
        doublematsigmoid(4,5,layer[1],layer[1]);
        matmul(4,5,5,6,layer[2],layer[1],weights[2]);
        matsum(4,6,layer[2],layer[2],bias[2]);
        doublematsigmoid(4,6,layer[2],layer[2]);
        matmul(4,6,6,6,layer[3],layer[2],weights[3]);
        matsum(4,6,layer[3],layer[3],bias[3]);
        doublematsigmoid(4,6,layer[3],layer[3]);
        matmul(4,6,6,6,layer[4],layer[3],weights[4]);
        matsum(4,6,layer[4],layer[4],bias[4]);
        doublematsigmoid(4,6,layer[4],layer[4]);
        matmul(4,6,6,4,layer[5],layer[4],weights[5]);
        matsum(4,4,layer[5],layer[5],bias[5]);
        doublematsigmoid(4,4,layer[5],layer[5]);
        matmul(4,5,5,1,output,layer[5],weights[6]);
        matsum(4,1,output,output,bias[6]);
        doublematsigmoid(4,1,output,output);
        printf("\nOutput\n");
        for (i=0;i<4;i++)
        {
            printf("%lf\t",output[i][0]);
        }
        printf("\n");
        /*for (i=0;i<4;i++)
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
                layer2[i][j]=sigmoid(product1+bias2[i][j]);
                product1=0;
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<6;j++)
            {
                for(k=0;k<5;k++)
                {
                    product1 = product1 + layer2[i][k]*weights3[k][j];
                }
                layer3[i][j]=LeakyRelu(product1+bias3[i][j]);
                product1=0;
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<6;j++)
            {
                for(k=0;k<6;k++)
                {
                    product1 = product1 + layer3[i][k]*weights4[k][j];
                }
                layer4[i][j]=LeakyRelu(product1+bias4[i][j]);
                product1=0;
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<6;j++)
            {
                for(k=0;k<6;k++)
                {
                    product1 = product1 + layer4[i][k]*weights5[k][j];
                }
                layer5[i][j]=LeakyRelu(product1+bias5[i][j]);
                product1=0;
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                for(k=0;k<6;k++)
                {
                    product1 = product1 + layer5[i][k]*weights6[k][j];
                }
                layer6[i][j]=sigmoid(product1+bias6[i][j]);
                product1=0;
            }
        }
        printf("\nOutput\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                product1 = product1 + layer6[i][j]*weights7[j];
            }
            output[i]=sigmoid(product1+bias7[i]);
            product1=0;
            printf("%f\t",output[i]);
        }
        printf("\n");*/
        
        
        /***********************BACKPROPAGATION****************************/
        for(i=0;i<4;i++)
        {
            interm1[i] = 2*(y[i] - output[i])*dsigmoid(output[i]);
        }
        mattranspose(4,4,tlayer[5],layer[5]);
        matmul(4,4,4,1,d_weights[6],tlayer[5],interm[0]);
        mattranspose(4,1,tweights[6],weights[6]);
        matmul(4,1,1,4,interm[1],interm[0],tweights[6]); 
        doublematdsigmoid(4,4,interm[1],layer[5]);
        
       
        mattranspose(4,6,tlayer[4],layer[4]);
        matmul(6,4,4,4,d_weights[5],tlayer[4],interm[1]);
        mattranspose(6,4,tweights[5],weights[5]);
        matmul(4,4,4,6,interm[2],interm[1],tweights[5]); 
        doublematdsigmoid(4,6,interm[2],layer[4]);
        
        mattranspose(4,6,tlayer[3],layer[3]);
        matmul(6,6,6,4,d_weights[4],tlayer[3],interm[2]);
        mattranspose(6,6,tweights[4],weights[4]);
        matmul(4,6,6,6,interm[3],interm[2],tweights[4]); 
        doublematdsigmoid(4,6,interm[3],layer[3]);
        
        mattranspose(4,6,tlayer[2],layer[2]);
        matmul(4,6,6,4,d_weights[3],tlayer[2],interm[3]);
        mattranspose(6,6,tweights[3],weights[3]);
        matmul(4,6,6,6,interm[4],interm[3],tweights[3]); 
        doublematdsigmoid(4,6,interm[2],layer[2]);
        
        mattranspose(4,5,tlayer[1],layer[1]);
        matmul(5,4,4,6,d_weights[2],tlayer[1],interm[3]);
        mattranspose(6,6,tweights[2],weights[2]);
        matmul(4,6,6,6,interm[4],interm[3],tweights[2]); 
        doublematdsigmoid(4,6,interm[2],layer[1]);
        
        matmul(4,4,4,5,d_weights2,tlayer1,interm2);
        mattranspose(4,5,tweights2,weights2);
        matmul(4,5,5,4,interm3,interm2,tweights2);
        doublematdsigmoid(4,5,interm3,layer1);
        mattranspose(4,6,tinput,input);
        matmul(3,4,4,4,d_weights1,tinput,interm3);
        /*for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                product1 = product1 + layer6[j][i]*interm1[j];
            }
            d_weights7[i]=product1;
            product1=0;
        }
        printf("\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                product1 = interm1[i]*weights7[j];
                interm2[i][j]=product1*dsigmoid(layer6[i][j]);
                product1=0;
            }
        }
        //printf("\n");
        //printf("\n");
        for (i=0;i<6;i++)
        {
            for(j=0;j<4;j++)
            {
                for(k=0;k<4;k++)
                {
                    product1 = product1 + layer5[k][i]*interm2[k][j];
                }
                d_weights6[i][j]=product1;
                product1=0;
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<6;j++)
            {
                for(k=0;k<4;k++)
                {
                product1 = product1 + interm2[i][k]*weights6[j][k];
                }
                interm3[i][j]=product1*dLeakyRelu(layer5[i][j]);
                product1=0;
            }
        }
        for (i=0;i<6;i++)
        {
            for(j=0;j<6;j++)
            {
                for(k=0;k<4;k++)
                {
                    product1 = product1 + layer4[k][i]*interm3[k][j];
                }
                d_weights5[i][j]=product1;
                product1=0;
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<6;j++)
            {
                for(k=0;k<6;k++)
                {
                product1 = product1 + interm3[i][k]*weights5[j][k];
                }
                interm4[i][j]=product1*dLeakyRelu(layer4[i][j]);
                product1=0;
            }
        }
        
        for (i=0;i<6;i++)
        {
            for(j=0;j<6;j++)
            {
                for(k=0;k<4;k++)
                {
                    product1 = product1 + layer3[k][i]*interm4[k][j];
                }
                d_weights4[i][j]=product1;
                product1=0;
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<6;j++)
            {
                for(k=0;k<6;k++)
                {
                product1 = product1 + interm4[i][k]*weights4[j][k];
                }
                interm5[i][j]=product1*dLeakyRelu(layer3[i][j]);
                product1=0;
            }
        }
        
        for (i=0;i<5;i++)
        {
            for(j=0;j<6;j++)
            {
                for(k=0;k<4;k++)
                {
                    product1 = product1 + layer2[k][i]*interm5[k][j];
                }
                d_weights3[i][j]=product1;
                product1=0;
            }
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                for(k=0;k<6;k++)
                {
                product1 = product1 + interm5[i][k]*weights3[j][k];
                }
                interm6[i][j]=product1*dsigmoid(layer2[i][j]);
                product1=0;
            }
        }
        
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
                for(k=0;k<4;k++)
                {
                    product1 = product1 + layer1[k][i]*interm6[k][j];
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
                product1 = product1 + interm6[i][k]*weights2[j][k];
                }
                interm7[i][j]=product1*dsigmoid(layer1[i][j]);
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
                    product1 = product1 + input[k][i]*interm7[k][j];
                }
                d_weights1[i][j]=product1;
                product1=0;
            }
        }*/
        
        
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
        for (i=0;i<5;i++)
        {
            for(j=0;j<6;j++)
            {
                weights3[i][j] +=alpha*d_weights3[i][j];
            }
        }
        for (i=0;i<6;i++)
        {
            for(j=0;j<6;j++)
            {
                weights4[i][j] +=alpha*d_weights4[i][j];
            }
        }
        for (i=0;i<6;i++)
        {
            for(j=0;j<6;j++)
            {
                weights5[i][j] +=alpha*d_weights5[i][j];
            }
        }
        for (i=0;i<6;i++)
        {
            for(j=0;j<4;j++)
            {
                weights6[i][j] +=alpha*d_weights6[i][j];
            }
        }
        for(j=0;j<4;j++)
        {
                weights7[j] +=alpha*d_weights7[j];
        }
        /*********************UPDATE BIAS*****************************/
          //      printf("\nBias1\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
               bias1[i][j] +=alpha*interm7[i][j];
               //printf("%f\t",bias1[i][j]);
            }
            //printf("\n");
        }
        //printf("\nBias2\n");
        for (i=0;i<4;i++)
        {
            for(j=0;j<5;j++)
            {
               bias2[i][j] +=alpha*interm6[i][j];
               //printf("%f\t",bias2[i][j]);
            }
            //printf("\n");
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<6;j++)
            {
               bias3[i][j] +=alpha*interm5[i][j];
               //printf("%f\t",bias2[i][j]);
            }
            //printf("\n");
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<6;j++)
            {
               bias4[i][j] +=alpha*interm4[i][j];
               //printf("%f\t",bias2[i][j]);
            }
            //printf("\n");
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<6;j++)
            {
               bias5[i][j] +=alpha*interm3[i][j];
               //printf("%f\t",bias2[i][j]);
            }
            //printf("\n");
        }
        for (i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
               bias6[i][j] +=alpha*interm2[i][j];
               //printf("%f\t",bias2[i][j]);
            }
            //printf("\n");
        }
        printf("\nBias7\n");
        for(j=0;j<4;j++)
        {
                bias7[j] +=alpha*interm1[j];
                printf("%f\t",bias7[j]);
        }
        printf("\n");
    }
}
