#include<stdio.h>
#include<time.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define NUM_LAYERS 8
#define INPUT_SIZE 2
int num_neurons[NUM_LAYERS]={INPUT_SIZE,5,10,10,20,10,4,1};
int rando() {
    int n;
    n = rand()%2 ;
    return n;
}

void doubledropout(int row, int col, double *ans[])
{
    int i,j;
    for(i=0;i<row;i++)
    {
        for(j=0;j<col;j++)
        {
             ans[i][j]=rando();
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
void doublefree(double ***var, int row)
{
    int i;
    for (int i=0; i<row; i++) 
    {
        free((*var)[i]);
    }
    free(*var);
}
void main()
{
    double **dphneurons[NUM_LAYERS-2];
    int i,j,dpnumneurons,epoch;
    srand(time(0));
    for(epoch=0;epoch<1;epoch++)
    {
        for(i=0;i<(NUM_LAYERS-2);i++)
        {
            doublemalloc(&dphneurons[i],1,num_neurons[i+1]);
            doubledropout(1,num_neurons[i+1],dphneurons[i]);
        }
        for(i=0;i<(NUM_LAYERS-2);i++)
        {
            dpnumneurons=0;
            for(j=0;j<num_neurons[i+1];j++)
            {
                printf("%lf  ",dphneurons[i][0][j]);
                if(dphneurons[i][0][j]>0)
                    dpnumneurons++;
            }
            printf("\n dpnumneurons : %d\n",dpnumneurons);
            if(dpnumneurons==0)
            {
                //printf("%d",epoch);
                doubledropout(1,num_neurons[i+1],dphneurons[i]);
            }
        }
        for(i=0;i<(NUM_LAYERS-2);i++)
        {
            doublefree(&dphneurons[i],1);
        }
    }

}
