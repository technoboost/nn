#include<stdio.h>
#include<time.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define NUM_LAYERS 8
#define BATCH_SIZE 4
#define INPUT_SIZE 2
#define CHECK_POINT 100
#define DATA_COUNT 20000
#define EPOCH_C 4000
int num_neurons[NUM_LAYERS]={INPUT_SIZE,5,10,10,20,10,4,1};
double y[BATCH_SIZE];
double alpha=0.001;
double beta_1 = 0.9;
double beta_2 = 0.999;
double epsilon = 1e-8;

int rando() {
    int n;
    n = rand()%2 ;
    return n;
}

void doubledropout(int row, int col, double *ans[], double *matrix[])
{
    int i,j;
    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
             ans[i][j]=rando()*matrix[i][j];
        }
    }
}
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
void storeWeights(char *fileName, double **weights[])
{
	FILE* stream = fopen(fileName, "w");
    
	int looper,j,k,size=0;
 	/*for(looper = 0; looper<NUM_LAYERS-1;looper++)
 	{
 		size = size+num_neurons[looper]*num_neurons[looper+1];
 	}
 	printf(" Total number of weights = %d",size);
 	*/
    
    for(looper =0;looper <NUM_LAYERS-1;looper++){
 		for(j = 0; j<num_neurons[looper];j++)
 		{
 			for(k = 0; k<num_neurons[looper+1];k++)
 			{
 				fprintf(stream,"%lf \n",weights[looper][j][k]);
 			}
 		}
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
void storeBias(char *fileName, double **bias[])
{
	FILE* stream = fopen(fileName, "w"); 
	int looper,j,k,size=0;
 	for(looper =0;looper <NUM_LAYERS-1;looper++){
 		for(j = 0; j<BATCH_SIZE;j++)
 		{
 			for(k = 0; k<num_neurons[looper+1];k++)
 			{
 				fprintf(stream,"%lf \n",bias[looper][j][k]);
 			}
 		}
 	}
    fclose(stream);
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
void storeT(char *fileName, long int t)
{
    FILE* stream = fopen(fileName, "w");
    fprintf(stream,"%ld \n",t);
    fclose(stream);
}
void restoreT(char *fileName, long int *t)
{
    int size;
    FILE* stream = fopen(fileName, "r");
    if (NULL != stream) {
        fseek (stream, 0, SEEK_END);
        size = ftell(stream);

        if (0 == size) {
            printf("file is empty\n");
        }
        else{
            fseek(stream, 0, SEEK_SET);
            fscanf(stream,"%ld \n",t);
        }
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
            //ans[i][j] +=alpha*params[i][j];
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
    int batch_iter=0;
    int i,j,k,epoch;
    long int t=0;
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
        doublezero(&bias[i],BATCH_SIZE,num_neurons[i+1]);
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
        t=0;
        
        restoreT("storeT.txt",&t);
        printf("t = :%ld\n",t);
        if(!t)
        {
            t=0;
        }
        else
        {
            printf("Restoring CHECK_POINT\n");
            restoreWeights("weights.txt", weights);
            restoreWeights("weights_m_t.txt", weights_m_t);
            restoreWeights("weights_v_t.txt", weights_v_t);
            restoreBias("bias.txt", bias);
            restoreBias("bias_m_t.txt", bias_m_t);
            restoreBias("bias_v_t.txt", bias_v_t);
        }
        epoch=t*BATCH_SIZE/DATA_COUNT;
        if(t<(EPOCH_C*DATA_COUNT/BATCH_SIZE))
        {
            while(epoch<EPOCH_C)
            {        
                if(epoch%CHECK_POINT==0)
                {
                    storeWeights("weights.txt", weights);
                    storeWeights("weights_m_t.txt", weights_m_t);
                    storeWeights("weights_v_t.txt", weights_v_t);
                    storeBias("bias.txt", bias);
                    storeBias("bias_m_t.txt", bias_m_t);
                    storeBias("bias_v_t.txt", bias_v_t);
                    storeT("storeT.txt",t);
                    printf("CHECK_POINT saved.\n");
                    printf("Epoch :%d\n",epoch);
                }
                linenumber = 0;
                while(linenumber<DATA_COUNT)
                {
                    
                    readcsvfile(argv[1],linenumber,BATCH_SIZE,layer[0]);
                    linenumber+=BATCH_SIZE;           
                    //printf("%d",linenumber);
                    for(batch_iter=0;batch_iter<1;batch_iter++)
                    {
                    
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
                        matmul(num_neurons[0],BATCH_SIZE,BATCH_SIZE,num_neurons[1],d_weights[0],tlayer[0],interm[NUM_LAYERS-2]);
                        
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
                    }//batch_iter
                }//while
                
                /*printf("\nOutput\n");
                for (i=0;i<BATCH_SIZE;i++)
                {
                    printf("%lf\t",output[i][0]);
                }
                printf("\n");*/
               epoch++;
            }//epoch
        }
        t=0;
        storeWeights("weights.txt", weights);
        storeWeights("weights_m_t.txt", weights_m_t);
        storeWeights("weights_v_t.txt", weights_v_t);
        storeBias("bias.txt", bias);
        storeBias("bias_m_t.txt", bias_m_t);
        storeBias("bias_v_t.txt", bias_v_t);
        storeT("storeT.txt",t);
        printf("CHECK_POINT saved.\n");
        /************************TESTING***************************************/
        readcsvfile(argv[2],0,4,layer[0]);
        for(i=0;i<(NUM_LAYERS-2);i++)
        {
            matmul(BATCH_SIZE,num_neurons[i],num_neurons[i],num_neurons[i+1],layer[i+1],layer[i],weights[i]);
            matsum(BATCH_SIZE,num_neurons[i+1],layer[i+1],layer[i+1],bias[i]);
            doublematsigmoid(4,num_neurons[i+1],layer[i+1],layer[i+1]);
        }
        matmul(BATCH_SIZE,num_neurons[i],num_neurons[i],num_neurons[i+1],output,layer[NUM_LAYERS-2],weights[NUM_LAYERS-2]);
        matsum(BATCH_SIZE,num_neurons[i+1],output,output,bias[NUM_LAYERS-2]);
        doublematsigmoid(BATCH_SIZE,num_neurons[i+1],output,output);
        printf("\nOutput\n");
        for (i=0;i<4;i++)
        {
            printf("%lf\t",output[i][0]);
        }
        printf("\n");
    }//else
}
