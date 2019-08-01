#include<stdio.h>
#include<time.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
double y[4]={0,0,0,0};
double alpha=0.1;
double beta_1 = 0.9;
double beta_2 = 0.999;
double epsilon = 1e-8;

const char* getfield(char* line, int num)
{
    const char* tok;
    for (tok = strtok(line, " ");
            tok && *tok;
            tok = strtok(NULL, " \n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}

void readcsvfile(char *inputfile,int linenumber,double *input[])
{
    FILE* stream = fopen(inputfile, "r");
    char line[1024];
    int i=0;
    while (fgets(line, 1024, stream))
    {
        if( i>=linenumber && i<(linenumber+4))
        {
            char* tmp = strdup(line);
            //printf("Field 1 would be %s", getfield(tmp, 1));
            sscanf(getfield(tmp,1), "%lf", &input[i-linenumber][0]);
            // NOTE strtok clobbers tmp
            free(tmp);
            tmp = strdup(line);
            
            sscanf(getfield(tmp,2), "%lf", &input[i-linenumber][1]);
            //printf(" %4.10lf\n",input[i-linenumber][1]);
            // NOTE strtok clobbers tmp
            free(tmp);
            input[i-linenumber][2]=1;
            tmp = strdup(line);
            
            sscanf(getfield(tmp,3), "%lf", &y[i-linenumber]);
            printf(" %lf\n",y[i-linenumber]);
            // NOTE strtok clobbers tmp
            free(tmp);
        }
        i++;
    }
    fclose(stream);
}
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
void doublezero(double ***var, int row, int col)
{
    int i,j;
    for (i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
            (*var)[i][j]=0 ;
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
void doublematleakyrelu(int row, int col,double *ans[], double *matrix[])
{
    int i,j;
    for(i=0; i<row; i++)
    {
        for(j=0; j<col; j++)
        {
            ans[i][j]=LeakyRelu(matrix[i][j]);
        }
    }
}
void doublematdleakyrelu(int row, int col,double *ans[], double *matrix[])
{
    int i,j;
    for(i=0; i<row; i++)
    {
        for(j=0; j<col; j++)
        {
            ans[i][j]*=dLeakyRelu(matrix[i][j]);
        }
    }
}
void updateparams(int row, int col, double *ans[], double *ans_m_t[], double *ans_v_t[], double *params[], int t)
{
    int i,j;
    double m_cap,v_cap;
    for (i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
            ans[i][j] +=alpha*params[i][j];
             ans_m_t[i][j] = beta_1*ans_m_t[i][j] + (1-beta_1)*params[i][j];	//updates the moving averages of the gradient
	            ans_v_t[i][j] = beta_2*ans_v_t[i][j] + (1-beta_2)*(params[i][j]*params[i][j]);	//updates the moving averages of the squared gradient
	            m_cap = ans_m_t[i][j]/(1-(pow(beta_1,t)));		//calculates the bias-corrected estimates
	            v_cap = ans_v_t[i][j]/(1-(pow(beta_2,t)));		//calculates the bias-corrected estimates
	            ans[i][j] = ans[i][j] + (alpha*m_cap)/(sqrt(v_cap)+epsilon);	//updates the parameters
        }
    }
}


void main(int argc, char *argv[])
{
    int linenumber = 0;
    int datacount = 500;
    int i,j,k,epoch,t;
    double **input;
    double **weights[7];
    double **weights_m_t[7];
    double **weights_v_t[7];
    double **output;
    double **layer[6];
    double **bias[7];
    double **bias_m_t[7];
    double **bias_v_t[7];
    double **d_weights[7];
    double **interm[7];
    double **tlayer[6];
    double **tinput;
    double **tweights[7];
    srand(time(0));
    doublemalloc(&weights[0],3,4);
    doublemalloc(&weights_m_t[0],3,4);
    doublemalloc(&weights_v_t[0],3,4);
    doublemalloc(&d_weights[0],3,4);
    doublerandom(&weights[0],3,4);
    doublezero(&weights_m_t[0],3,4);
    doublezero(&weights_v_t[0],3,4);
    
    
    doublemalloc(&weights[1],4,5);
    doublemalloc(&weights_m_t[1],4,5);
    doublemalloc(&weights_v_t[1],4,5);
    doublemalloc(&d_weights[1],4,5);
    doublerandom(&weights[1],4,5);
    doublezero(&weights_m_t[1],4,5);
    doublezero(&weights_v_t[1],4,5);
    
    
    doublemalloc(&weights[2],5,6);
    doublemalloc(&weights_m_t[2],5,6);
    doublemalloc(&weights_v_t[2],5,6);
    doublemalloc(&d_weights[2],5,6);
    doublerandom(&weights[2],5,6);
    doublezero(&weights_m_t[2],5,6);
    doublezero(&weights_v_t[2],5,6);
    
    
    doublemalloc(&weights[3],6,6);
    doublemalloc(&weights_m_t[3],6,6);
    doublemalloc(&weights_v_t[3],6,6);
    doublemalloc(&d_weights[3],6,6);
    doublerandom(&weights[3],6,6);
    doublezero(&weights_m_t[3],6,6);
    doublezero(&weights_v_t[3],6,6);
    
    doublemalloc(&weights[4],6,6);
    doublemalloc(&weights_m_t[4],6,6);
    doublemalloc(&weights_v_t[4],6,6);
    doublemalloc(&d_weights[4],6,6);
    doublerandom(&weights[4],6,6);
    doublezero(&weights_m_t[4],6,6);
    doublezero(&weights_v_t[4],6,6);
    
    doublemalloc(&weights[5],6,4);
    doublemalloc(&weights_m_t[5],6,4);
    doublemalloc(&weights_v_t[5],6,4);
    doublemalloc(&d_weights[5],6,4);
    doublerandom(&weights[5],6,4);
    doublezero(&weights_m_t[5],6,4);
    doublezero(&weights_v_t[5],6,4);
    
    doublemalloc(&weights[6],4,1);
    doublemalloc(&weights_m_t[6],4,1);
    doublemalloc(&weights_v_t[6],4,1);
    doublemalloc(&d_weights[6],4,1);
    doublerandom(&weights[6],4,1);
    doublezero(&weights_m_t[6],4,1);
    doublezero(&weights_v_t[6],4,1);
    
    doublemalloc(&bias[0],4,4);
    doublezero(&bias[0],4,4);
    doublemalloc(&bias[1],4,5);
    doublezero(&bias[1],4,5);
    doublemalloc(&bias[2],4,6);
    doublezero(&bias[2],4,6);
    doublemalloc(&bias[3],4,6);
    doublezero(&bias[3],4,6);
    doublemalloc(&bias[4],4,6);
    doublezero(&bias[4],4,6);
    doublemalloc(&bias[5],4,4);
    doublezero(&bias[5],4,4);
    doublemalloc(&bias[6],4,1);
    doublezero(&bias[6],4,1);
    
    doublemalloc(&bias_m_t[0],4,4);
    doublemalloc(&bias_v_t[0],4,4);
    doublemalloc(&bias_m_t[1],4,5);
    doublemalloc(&bias_v_t[1],4,5);
    doublemalloc(&bias_m_t[2],4,6);
    doublemalloc(&bias_v_t[2],4,6);
    doublemalloc(&bias_m_t[3],4,6);
    doublemalloc(&bias_v_t[3],4,6);
    doublemalloc(&bias_m_t[4],4,6);
    doublemalloc(&bias_v_t[4],4,6);
    doublemalloc(&bias_m_t[5],4,4);
    doublemalloc(&bias_v_t[5],4,4);
    doublemalloc(&bias_m_t[6],4,1);
    doublemalloc(&bias_v_t[6],4,1);
    
    doublezero(&bias_m_t[0],4,4);
    doublezero(&bias_v_t[0],4,4);
    doublezero(&bias_m_t[1],4,5);
    doublezero(&bias_v_t[1],4,5);
    doublezero(&bias_m_t[2],4,6);
    doublezero(&bias_v_t[2],4,6);
    doublezero(&bias_m_t[3],4,6);
    doublezero(&bias_v_t[3],4,6);
    doublezero(&bias_m_t[4],4,6);
    doublezero(&bias_v_t[4],4,6);
    doublezero(&bias_m_t[5],4,4);
    doublezero(&bias_v_t[5],4,4);
    doublezero(&bias_m_t[6],4,1);
    doublezero(&bias_v_t[6],4,1);

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
    doublemalloc(&tweights[0],4,3);
    doublemalloc(&tweights[1],5,4);
    doublemalloc(&tweights[2],6,5);
    doublemalloc(&tweights[3],6,6);
    doublemalloc(&tweights[4],6,6);
    doublemalloc(&tweights[5],4,6);
    doublemalloc(&tweights[6],1,4);
    doublemalloc(&tinput,3,4);
    if (argc>2)
    {
        printf("\n Sorry wrong arguments\n");
    }
    else if(argc==1)
    {
        printf("\n Sorry no file provided as argument\n");
    }
    else
    {
        while(linenumber<datacount)
        {
            readcsvfile(argv[1],linenumber,input);
            linenumber+=4;           
            printf("%d",linenumber);
            t=0;
            for(epoch=0;epoch<100;epoch++)
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
                doublematleakyrelu(4,6,layer[2],layer[2]);
                matmul(4,6,6,6,layer[3],layer[2],weights[3]);
                matsum(4,6,layer[3],layer[3],bias[3]);
                doublematleakyrelu(4,6,layer[3],layer[3]);
                matmul(4,6,6,6,layer[4],layer[3],weights[4]);
                matsum(4,6,layer[4],layer[4],bias[4]);
                doublematsigmoid(4,6,layer[4],layer[4]);
                matmul(4,6,6,4,layer[5],layer[4],weights[5]);
                matsum(4,4,layer[5],layer[5],bias[5]);
                doublematsigmoid(4,4,layer[5],layer[5]);
                matmul(4,4,4,1,output,layer[5],weights[6]);
                matsum(4,1,output,output,bias[6]);
                doublematsigmoid(4,1,output,output);
                
                /***********************BACKPROPAGATION****************************/
                for(i=0;i<4;i++)
                {
                    interm[0][i][0] = 2*(y[i] - output[i][0])*dsigmoid(output[i][0]);
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
                matmul(6,4,4,6,d_weights[4],tlayer[3],interm[2]);
                mattranspose(6,6,tweights[4],weights[4]);
                matmul(4,6,6,6,interm[3],interm[2],tweights[4]); 
                doublematdleakyrelu(4,6,interm[3],layer[3]);
                
                mattranspose(4,6,tlayer[2],layer[2]);
                matmul(6,4,4,6,d_weights[3],tlayer[2],interm[3]);
                mattranspose(6,6,tweights[3],weights[3]);
                matmul(4,6,6,6,interm[4],interm[3],tweights[3]); 
                doublematdleakyrelu(4,6,interm[4],layer[2]);
                
                mattranspose(4,5,tlayer[1],layer[1]);
                matmul(5,4,4,6,d_weights[2],tlayer[1],interm[4]);
                mattranspose(5,6,tweights[2],weights[2]);
                matmul(4,6,6,5,interm[5],interm[4],tweights[2]); 
                doublematdsigmoid(4,5,interm[5],layer[1]);
                
                mattranspose(4,4,tlayer[0],layer[0]);
                matmul(4,4,4,5,d_weights[1],tlayer[0],interm[5]);
                mattranspose(4,5,tweights[1],weights[1]);
                matmul(4,5,5,4,interm[6],interm[5],tweights[1]); 
                doublematdsigmoid(4,4,interm[6],layer[0]);
                
                mattranspose(4,3,tinput,input);
                matmul(3,4,4,4,d_weights[0],tinput,interm[6]);
                
                /*********************UPDATE WEIGHTS*****************************/
                t+=1;
                updateparams(3,4,weights[0],weights_m_t[0],weights_v_t[0],d_weights[0],t);
                updateparams(4,5,weights[1],weights_m_t[1],weights_v_t[1],d_weights[1],t);
                updateparams(5,6,weights[2],weights_m_t[2],weights_v_t[2],d_weights[2],t);
                updateparams(6,6,weights[3],weights_m_t[3],weights_v_t[3],d_weights[3],t);
                updateparams(6,6,weights[4],weights_m_t[4],weights_v_t[4],d_weights[4],t);
                updateparams(6,4,weights[5],weights_m_t[5],weights_v_t[5],d_weights[5],t);
                updateparams(4,1,weights[6],weights_m_t[6],weights_v_t[6],d_weights[6],t);
                
                /*********************UPDATE BIAS*****************************/
                updateparams(4,4,bias[0],bias_m_t[0],bias_v_t[0],interm[6],t);
                updateparams(4,5,bias[1],bias_m_t[1],bias_v_t[1],interm[5],t);
                updateparams(4,6,bias[2],bias_m_t[2],bias_v_t[2],interm[4],t);
                updateparams(4,6,bias[3],bias_m_t[3],bias_v_t[3],interm[3],t);
                updateparams(4,6,bias[4],bias_m_t[4],bias_v_t[4],interm[2],t);
                updateparams(4,4,bias[5],bias_m_t[5],bias_v_t[5],interm[1],t);
                updateparams(4,1,bias[6],bias_m_t[6],bias_v_t[6],interm[0],t);
            }//epoch
            printf("\nOutput\n");
                for (i=0;i<4;i++)
                {
                    printf("%lf\t",output[i][0]);
                }
                printf("\n");
        }//while
    }//else
}
