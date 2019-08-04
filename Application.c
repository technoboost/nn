#include<stdio.h>
#include<time.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define NUM_LAYERS 8
#define BATCH_SIZE 4
#define INPUT_SIZE 2
#define TEST_SIZE 10000
double y[BATCH_SIZE];

int num_neurons[NUM_LAYERS]={INPUT_SIZE,5,10,10,20,10,4,1};
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
                sscanf(getfield(tmp,j+1), "%lf", &input[i-linenumber][j]);
                // NOTE strtok clobbers tmp
                free(tmp);
            }
            tmp = strdup(line);
            sscanf(getfield(tmp,INPUT_SIZE+1), "%lf", &y[i-linenumber]);
            //printf(" %lf\n",y[i-linenumber]);
            // NOTE strtok clobbers tmp
            free(tmp);
        }
        i++;
    }
    fclose(stream);
}
void restoreWeights(char *fileName, double **weights[])
{
    FILE* stream = fopen(fileName, "r");
    if (stream == NULL) 
    { 
        printf("Could not open file %s\n", fileName); 
        return ; 
    } 
    int looper,j,k;
    for(looper =0;looper <NUM_LAYERS-1;looper++){
        for(j = 0; j<num_neurons[looper];j++)
        {
            for(k = 0; k<num_neurons[looper+1];k++)
            {
                fscanf(stream,"%lf",&weights[looper][j][k]);
            }
        }
    }
    fclose(stream);
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
void restoreBias(char *fileName, double **bias[])
{
    FILE* stream = fopen(fileName, "r");
    if (stream == NULL) 
    { 
        printf("Could not open file %s\n", fileName); 
        return ; 
    } 
    int looper,j,k,size=0;
    for(looper =0;looper <NUM_LAYERS-1;looper++){
        for(j = 0; j<BATCH_SIZE;j++)
        {
            for(k = 0; k<num_neurons[looper+1];k++)
            {
                fscanf(stream,"%lf \n",&bias[looper][j][k]);
            }
        }
    }
    fclose(stream);
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
void doublemalloc(double ***var,int row,int col)
{
    int i;
    *var = (double **)malloc(row * sizeof(double *)); 
        for (i=0; i<row; i++) 
        {
             (*var)[i] = (double *)malloc(col * sizeof(double)); 
        }
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

void main(int argc, char *argv[])
{
    double **weights[NUM_LAYERS-1];
    double **bias[NUM_LAYERS-1];
    double **layer[NUM_LAYERS-1];
    double **output;
    int i,j,k;
    double fv=0;
 
    for(i=0;i<(NUM_LAYERS-1);i++)
    {
        doublemalloc(&weights[i],num_neurons[i],num_neurons[i+1]);
        doublemalloc(&bias[i],BATCH_SIZE,num_neurons[i+1]);
        doublemalloc(&layer[i],BATCH_SIZE,num_neurons[i]);
    }
    doublemalloc(&output,BATCH_SIZE,1);
    restoreWeights("weights.txt", weights);
    restoreBias("bias.txt", bias);
    for (j=0;j<TEST_SIZE;j+=4)
    {
        readcsvfile(argv[1],j,BATCH_SIZE,layer[0]);
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
        for (i=0;i<BATCH_SIZE;i++)
        {
            printf("%lf\t",output[i][0]);
            if(output[i][0]>0.5)
            {
                output[i][0]=1;
            }
            else
            {
                 output[i][0]=0;
            }
            if(output[i][0]!=y[i])
            {
                //printf("%lf\t",layer[0][i][0]);
                //printf("%lf\t",y[i]);
                fv++;
            }
        }
         printf("\n");
    }
    printf("Error rate : %f \n",(double)(fv/TEST_SIZE));
    printf("Errors : %f \n",fv);
}
