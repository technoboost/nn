#include<stdio.h>
#include<time.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define NUM_LAYERS 8
#define BATCH_SIZE 4
#define INPUT_SIZE 2
int num_neurons[NUM_LAYERS]={INPUT_SIZE,6,10,20,10,6,4,1};
double y[4]={0,0,0,0};
double alpha=0.01;
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

void readcsvfile(char *inputfile,int linenumber,int size, double *input[])
{
    FILE* stream = fopen(inputfile, "r");
    char line[1024];
    int i=0,j;
    char* tmp;
    while (fgets(line, 1024, stream))
    {
        if( i>=linenumber && i<(linenumber+size))
        {
            for(j=0;j<INPUT_SIZE;j++)
            { 
                tmp = strdup(line);
                //printf("Field 1 would be %s", getfield(tmp, 1));
                sscanf(getfield(tmp,1), "%lf", &input[i-linenumber][j]);
                // NOTE strtok clobbers tmp
                free(tmp);
            }
            tmp = strdup(line);
            sscanf(getfield(tmp,3), "%lf", &y[i-linenumber]);
            //printf(" %lf\n",y[i-linenumber]);
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
             /*ans_m_t[i][j] = beta_1*ans_m_t[i][j] + (1-beta_1)*params[i][j];	//updates the moving averages of the gradient
	         ans_v_t[i][j] = beta_2*ans_v_t[i][j] + (1-beta_2)*(params[i][j]*params[i][j]);	//updates the moving averages of the squared gradient
	         m_cap = ans_m_t[i][j]/(1-(pow(beta_1,t)));		//calculates the bias-corrected estimates
	         v_cap = ans_v_t[i][j]/(1-(pow(beta_2,t)));		//calculates the bias-corrected estimates
	         ans[i][j] = ans[i][j] + (alpha*m_cap)/(sqrt(v_cap)+epsilon);*/	//updates the parameters
        }
    }
}


void main(int argc, char *argv[])
{
    int linenumber = 0;
    int datacount = 500;
    int i,j,k,epoch,t;
    double **weights[NUM_LAYERS-1];
    double **weights_m_t[NUM_LAYERS-1];
    double **weights_v_t[NUM_LAYERS-1];
    double **output;
    double **layer[NUM_LAYERS-1];
    double **bias[NUM_LAYERS-1];
    double **bias_m_t[NUM_LAYERS-1];
    double **bias_v_t[NUM_LAYERS-1];
    double **d_weights[NUM_LAYERS-1];
    double **interm[NUM_LAYERS-1];
    double **tlayer[NUM_LAYERS-1];
    double **tweights[NUM_LAYERS-1];
    srand(time(0));
    for(i=0;i<(NUM_LAYERS-1);i++)
    {
        doublemalloc(&weights[i],num_neurons[i],num_neurons[i+1]);
        doublemalloc(&weights_m_t[i],num_neurons[i],num_neurons[i+1]);
        doublemalloc(&weights_v_t[i],num_neurons[i],num_neurons[i+1]);
        doublemalloc(&d_weights[i],num_neurons[i],num_neurons[i+1]);
        doublerandom(&weights[i],num_neurons[i],num_neurons[i+1]);
        doublezero(&weights_m_t[i],num_neurons[i],num_neurons[i+1]);
        doublezero(&weights_v_t[i],num_neurons[i],num_neurons[i+1]);
        doublemalloc(&tweights[i],num_neurons[i+1],num_neurons[i]);
        doublemalloc(&bias[i],BATCH_SIZE,num_neurons[i+1]);
        doublerandom(&bias[i],BATCH_SIZE,num_neurons[i+1]);
        doublemalloc(&bias_m_t[i],BATCH_SIZE,num_neurons[i+1]);
        doublemalloc(&bias_v_t[i],BATCH_SIZE,num_neurons[i+1]);
        doublezero(&bias_m_t[i],BATCH_SIZE,num_neurons[i+1]);
        doublezero(&bias_v_t[i],BATCH_SIZE,num_neurons[i+1]);
        doublemalloc(&interm[NUM_LAYERS-i-2],BATCH_SIZE,num_neurons[i+1]);
        doublemalloc(&layer[i],BATCH_SIZE,num_neurons[i]);
        doublemalloc(&tlayer[i],num_neurons[i],BATCH_SIZE);
    }
    doublemalloc(&output,BATCH_SIZE,1);
    if (argc>3)
    {
        printf("\n Sorry wrong arguments\n");
    }
    else if(argc==1)
    {
        printf("\n Sorry no file provided as argument\n");
    }
    else
    {
        for(epoch=0;epoch<100;epoch++)
        {
            linenumber = 0;
            printf("%d",epoch);
            while(linenumber<datacount)
            {
                
                readcsvfile(argv[1],linenumber,BATCH_SIZE,layer[0]);
                linenumber+=BATCH_SIZE;           
                //printf("%d",linenumber);
                t=0;
            
                
                /***********************FEEDFORWARD**************************/
                for(i=0;i<(NUM_LAYERS-2);i++)
                {
                    matmul(BATCH_SIZE,num_neurons[i],num_neurons[i],num_neurons[i+1],layer[i+1],layer[i],weights[i]);
                    matsum(BATCH_SIZE,num_neurons[i+1],layer[i+1],layer[i+1],bias[i]);
                    doublematsigmoid(BATCH_SIZE,num_neurons[i+1],layer[i+1],layer[i+1]);
                }
                matmul(BATCH_SIZE,num_neurons[i],num_neurons[i],num_neurons[i+1],output,layer[NUM_LAYERS-2],weights[NUM_LAYERS-2]);
                matsum(BATCH_SIZE,num_neurons[i+1],output,output,bias[NUM_LAYERS-2]);
                doublematsigmoid(BATCH_SIZE,num_neurons[i+1],output,output);
                
                /***********************BACKPROPAGATION****************************/
                for(i=0;i<BATCH_SIZE;i++)
                {
                    interm[0][i][0] = 2*(y[i] - output[i][0])*dsigmoid(output[i][0]);
                }
                for(i=0;i <(NUM_LAYERS-2);i++)
                {
                    mattranspose(BATCH_SIZE,num_neurons[NUM_LAYERS-2-i],tlayer[NUM_LAYERS-2-i],layer[NUM_LAYERS-2-i]);
                    matmul(num_neurons[NUM_LAYERS-2-i],BATCH_SIZE,BATCH_SIZE,num_neurons[NUM_LAYERS-1-i],d_weights[NUM_LAYERS-2-i],tlayer[NUM_LAYERS-2-i],interm[i]);
                    mattranspose(num_neurons[NUM_LAYERS-2-i],num_neurons[NUM_LAYERS-1-i],tweights[NUM_LAYERS-2-i],weights[NUM_LAYERS-2-i]);
                    matmul(BATCH_SIZE,num_neurons[NUM_LAYERS-1-i],num_neurons[NUM_LAYERS-1-i],num_neurons[NUM_LAYERS-2-i],interm[i+1],interm[i],tweights[NUM_LAYERS-2-i]); 
                    doublematdsigmoid(BATCH_SIZE,num_neurons[NUM_LAYERS-2-i],interm[i+1],layer[NUM_LAYERS-2-i]);
                }               
                mattranspose(BATCH_SIZE,num_neurons[0],tlayer[0],layer[0]);
                matmul(num_neurons[0],BATCH_SIZE,BATCH_SIZE,num_neurons[1],d_weights[0],tlayer[0],interm[6]);
                
                /*********************UPDATE WEIGHTS*****************************/
                t+=1;
                for(i=0;i <(NUM_LAYERS-1);i++)
                {
                    updateparams(num_neurons[i],num_neurons[i+1],weights[i],weights_m_t[i],weights_v_t[i],d_weights[i],t);
                }
                
                
                /*********************UPDATE BIAS*****************************/
                for(i=0;i <(NUM_LAYERS-1);i++)
                {
                    updateparams(BATCH_SIZE,num_neurons[i+1],bias[i],bias_m_t[i],bias_v_t[i],interm[NUM_LAYERS-2-i],t);
                }
            }//while
            printf("\nOutput\n");
                for (i=0;i<BATCH_SIZE;i++)
                {
                    printf("%lf\t",output[i][0]);
                }
                printf("\n");
           
        }//epoch
                /************************TESTING***************************************/
                readcsvfile(argv[2],0,1,layer[0]);
                for(i=0;i<(NUM_LAYERS-2);i++)
                {
                    matmul(BATCH_SIZE,num_neurons[i],num_neurons[i],num_neurons[i+1],layer[i+1],layer[i],weights[i]);
                    matsum(BATCH_SIZE,num_neurons[i+1],layer[i+1],layer[i+1],bias[i]);
                    doublematsigmoid(BATCH_SIZE,num_neurons[i+1],layer[i+1],layer[i+1]);
                }
                matmul(BATCH_SIZE,num_neurons[i],num_neurons[i],num_neurons[i+1],output,layer[NUM_LAYERS-2],weights[NUM_LAYERS-2]);
                matsum(BATCH_SIZE,num_neurons[i+1],output,output,bias[NUM_LAYERS-2]);
                doublematsigmoid(BATCH_SIZE,num_neurons[i+1],output,output);
                printf("\nOutput\n");
                for (i=0;i<1;i++)
                {
                    printf("%lf\t",output[i][0]);
                }
                printf("\n");
    }//else
}
