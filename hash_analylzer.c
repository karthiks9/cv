#include <stdio.h>
#include <errno.h>
#include "vision.h"

#define HASH_DATA "hash_file"

#define IMAGE_ELEMENTS_TOTAL (IMAGE_NUM_PIXELS * IMAGE_NUM_PIXELS)
#define V1S_ELEMENTS_TOTAL (V1S_NUM_CHANNELS * V1S_NUM_ROWS * V1S_NUM_COLS)
#define V1C_ELEMENTS_TOTAL (V1C_NUM_CHANNELS * V1C_NUM_ROWS * V1C_NUM_COLS)
#define V2S_ELEMENTS_TOTAL (V2S_NUM_CHANNELS * V2S_NUM_ROWS * V2S_NUM_COLS)
#define V2C_ELEMENTS_TOTAL (V2C_NUM_CHANNELS * V2C_NUM_ROWS * V2C_NUM_COLS)
#define V4S_ELEMENTS_TOTAL (V4S_NUM_CHANNELS * V4S_NUM_ROWS * V4S_NUM_COLS)

#define TOTAL_ELEMENTS (IMAGE_ELEMENTS_TOTAL + V1S_ELEMENTS_TOTAL + V1C_ELEMENTS_TOTAL + V2S_ELEMENTS_TOTAL + V2C_ELEMENTS_TOTAL + V4S_ELEMENTS_TOTAL)

struct hash_element {
	int learned;
	float hash[TOTAL_ELEMENTS];
};

struct hash_element worldhash[10]; //for 10 digits

int get_num_elements(int i)
{
	int num_elements = 0;
	int j;

	for(j=0; j<TOTAL_ELEMENTS; j++) {
		if(worldhash[i].hash[j] != 0) {
			num_elements++;
		}
	}

	return num_elements;
}

int main(int argc, char *argv[])
{
	int i;
	int j;
	FILE *fptr;
	int num = 0;
	int num_elements = 0;
	
	fptr = fopen(HASH_DATA, "rb");

	if(!fptr) {
		printf("Hash file not found\n");
		return 0;
	}

	num = fread(worldhash, sizeof(struct hash_element), 10, fptr);

	if(!num) {
		printf("fread returned 0: %d\n", errno);
		return 0;
	}

	for(i=0; i<10; i++) {
		printf("Digit: %d\n", i);
		printf("Learning Done: %d\n", worldhash[i].learned);
		num_elements = get_num_elements(i);
		printf("number of elements set: %d\n", num_elements);
		for(j=0; j<TOTAL_ELEMENTS; j++) {
			if(worldhash[i].hash[j] != 0.0) {
				printf("index: %d hash: %f\n", j, worldhash[i].hash[j]);
			}
		}
	}
}
