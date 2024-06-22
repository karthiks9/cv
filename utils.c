#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "vision.h"

float mat_mult(float feature1[V2C_NUM_ROWS][V2C_NUM_COLS], float feature2[V2C_NUM_ROWS][V2C_NUM_COLS])
{
        float output = 0.0;
        float value1, value2;
        int i,k;

        for(i=0; i<V2C_NUM_ROWS; i++) {

                for(k=0; k<V2C_NUM_COLS; k++) {

                        /* multiply corresponding positions in the matrices */
                        value1 = feature1[i][k];
                        value2 = feature2[i][k];

                        output += value1 * value2;
                }
        }
        return output;
}

/* prints a float matrix given its dimensions */
void print_matrix(void *input, int num_rows, int num_cols)
{
        int i, j, col_len = num_cols * sizeof(float);
	float value;
	char *input_ptr = (char *)input;
	char *input_ptr2;

        for(i=0; i<num_rows; i++) {
                for(j=0; j<num_cols; j++) {
			input_ptr2 = input_ptr + (i * col_len) + (j * sizeof(float));
			value = *(float *)(input_ptr2);
                        printf("%0.0f ", value);
                }
                printf("\n");
        }
        printf("\n");
}

void print_cones(unsigned char cones[CONES_NUM_ROWS][CONES_NUM_COLS])
{
        int i, j;

        for(i=0; i<CONES_NUM_ROWS; i++) {
                for(j=0; j<CONES_NUM_COLS; j++) {
                        printf("%u ", cones[i][j]);
                }
                printf("\n");
        }
        printf("\n");
        return;
}

void print_image(unsigned char image[IMAGE_NUM_PIXELS][IMAGE_NUM_PIXELS])
{
        int i, j;

        for(i=0; i<IMAGE_NUM_PIXELS; i++) {
                for(j=0; j<IMAGE_NUM_PIXELS; j++) {
                        printf("%u ", image[i][j]);
                }
                printf("\n");
        }
        printf("\n");
        return;
}

void print_filter(float filter[CS_NUM_ROWS][CS_NUM_COLS])
{
	print_matrix(filter, CS_NUM_ROWS, CS_NUM_COLS);
}

void print_rgc(float rgc_cells[RGC_NUM_ROWS][RGC_NUM_COLS])
{
	print_matrix(rgc_cells, RGC_NUM_ROWS, RGC_NUM_COLS);
}

void print_v1s(float v1s[V1S_NUM_CHANNELS][V1S_NUM_ROWS][V1S_NUM_COLS])
{
	int c;

	for(c=0; c<V1S_NUM_CHANNELS; c++) {
		printf("Channel: %d\n", c);
		print_matrix(v1s[c], V1S_NUM_ROWS, V1S_NUM_COLS);
		printf("\n");
	}
}

void print_v2s(float v2s[V2S_NUM_CHANNELS][V2S_NUM_ROWS][V2S_NUM_COLS])
{
	int i;

	for(i=0; i<V2S_NUM_CHANNELS; i++) {
		printf("Channel: %d\n\n", i);
		print_matrix(v2s[i], V2S_NUM_ROWS, V2S_NUM_COLS);
		printf("\n");
	}
}

void print_v1c(float v1c[V1C_NUM_CHANNELS][V1C_NUM_ROWS][V1C_NUM_COLS])
{
	int c;

	for(c=0; c<V1C_NUM_CHANNELS; c++) {
		printf("Channel: %d\n\n", c);
		print_matrix(v1c[c], V1C_NUM_ROWS, V1C_NUM_COLS);
		printf("\n");
	}
}

void print_v2c(float v2c[V2C_NUM_CHANNELS][V2C_NUM_ROWS][V2C_NUM_COLS])
{
	int c;

	for(c=0; c<V2C_NUM_CHANNELS; c++) {
		printf("Channel: %d\n\n", c);
		print_matrix(v2c[c], V2C_NUM_ROWS, V2C_NUM_COLS);
		printf("\n");
	}
}

void print_v4s(float v4s[V4S_NUM_CHANNELS][V4S_NUM_ROWS][V4S_NUM_COLS])
{
	int c;

	for(c=0; c<V4S_NUM_CHANNELS; c++) {
		printf("Channel: %d\n", c);
		print_matrix(v4s[c], V4S_NUM_ROWS, V4S_NUM_COLS);
		printf("\n");
	}
}

float relu(float input)
{
	/* return max(input, 0.0) */

	if(input > 0.0)
		return input;
	else
		return 0.0;
}
