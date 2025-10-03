#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vision.h"
#include "utils.h"

#define BKUP_FILES_DIR_NAME "bkup_files"
#define V4S_FILTER_FILE_NAME "v4_filter_file"

#define HASH_FILE_NAME "hash_file"

#define IMAGE_ELEMENTS_TOTAL (IMAGE_NUM_PIXELS * IMAGE_NUM_PIXELS)
#define V1S_ELEMENTS_TOTAL (V1S_NUM_CHANNELS * V1S_NUM_ROWS * V1S_NUM_COLS)
#define V1C_ELEMENTS_TOTAL (V1C_NUM_CHANNELS * V1C_NUM_ROWS * V1C_NUM_COLS)
#define V2S_ELEMENTS_TOTAL (V2S_NUM_CHANNELS * V2S_NUM_ROWS * V2S_NUM_COLS)
#define V2C_ELEMENTS_TOTAL (V2C_NUM_CHANNELS * V2C_NUM_ROWS * V2C_NUM_COLS)
#define V4S_ELEMENTS_TOTAL (V4S_NUM_CHANNELS * V4S_NUM_ROWS * V4S_NUM_COLS)

#define TOTAL_ELEMENTS (IMAGE_ELEMENTS_TOTAL + V1S_ELEMENTS_TOTAL + V1C_ELEMENTS_TOTAL + V2S_ELEMENTS_TOTAL + V2C_ELEMENTS_TOTAL + V4S_ELEMENTS_TOTAL)

#define IMAGE_THRESHOLD 64
#define V1S_THRESHOLD 400
#define V1C_THRESHOLD 400
#define V2S_THRESHOLD 400000
#define V2C_THRESHOLD 400000
#define V4S_THRESHOLD (2 * pow(10,10)) 

unsigned char image[IMAGE_NUM_PIXELS] [IMAGE_NUM_PIXELS];
unsigned char cones[CONES_NUM_ROWS] [CONES_NUM_COLS];
float rgc_cells[RGC_NUM_ROWS] [RGC_NUM_COLS];
float v1s[V1S_NUM_CHANNELS][V1S_NUM_ROWS][V1S_NUM_COLS];
float v1c[V1C_NUM_CHANNELS][V1C_NUM_ROWS][V1C_NUM_COLS];
float v2s[V2S_NUM_CHANNELS][V2S_NUM_ROWS][V2S_NUM_COLS];
float v2c[V2C_NUM_CHANNELS][V2C_NUM_ROWS][V2C_NUM_COLS];

float v4s[V4S_NUM_CHANNELS][V4S_NUM_ROWS][V4S_NUM_COLS];
float v4s_filters[V4S_FILTER_NUM_CHANNELS][V4S_FILTER_ROWS][V4S_FILTER_COLS];

#define INITIAL_HASH_VALUE (0.25)

typedef struct world_state {
	int learned; /* has this state already been learned */
	float hash[TOTAL_ELEMENTS];
	//unsigned short positions[TOTAL_ELEMENTS]; //this may not be needed
	//int num_positions;
} world_state_t;

unsigned char hashcode[TOTAL_ELEMENTS];
unsigned short positions[TOTAL_ELEMENTS];
int num_positions = 0;
unsigned int train_mode = 1;
world_state_t world_state[10]; //world_state contains hash for each digit

int label; /* label used during trainning mode */

//#ifdef OLD_FILTERS //this is old and gold  ..
#if 1
float v1_filters[V1_NUM_FILTERS][V1_FILTER_ROWS][V1_FILTER_COLS] = {
                                                                    {{1,1,1}, /* 180 horizontal */
                                                                     {0,0,0},
                                                                     {0,0,0}},

                                                                    {{0,0,0},  /* 155 */
                                                                     {1,0,0},
                                                                     {0,1,0}},

                                                                    {{1,0,0},  /* 135 */
                                                                     {0,1,0},
                                                                     {0,0,1}},

                                                                    {{0,1,0},  /* 115 */
                                                                     {0,0,1},
                                                                     {0,0,0}},

                                                                    {{1,0,0},  /* 90 */
                                                                     {1,0,0},
                                                                     {1,0,0}},

                                                                    {{0,1,0},  /* 65 */
                                                                     {1,0,0},
                                                                     {0,0,0}},

                                                                    {{0,0,1},  /* 45 */
                                                                     {0,1,0},
                                                                     {1,0,0}},

                                                                    {{0,0,0},  /* 25 */
                                                                     {0,0,1},
                                                                     {0,1,0}},

                                                                    {{0,0,0},  /* 0 */
                                                                     {0,0,0},
                                                                     {1,1,1}}
                                                                   };
#else 
//this is experimental
float v1_filters[V1_NUM_FILTERS][V1_FILTER_ROWS][V1_FILTER_COLS] = {
                                                                    {{1,1,1}, /* 180 horizontal */
                                                                     {0,0,0},
                                                                     {-1,-1,-1}},

                                                                    {{0,-1,0},  /* 155 */
                                                                     {1,0,-1},
                                                                     {0,1,0}},

                                                                    {{1,0,0},  /* 135 */
                                                                     {0,1,0},
                                                                     {0,0,1}},

                                                                    {{0,1,0},  /* 115 */
                                                                     {-1,0,1},
                                                                     {0,-1,0}},

                                                                    {{1,0,-1},  /* 90 */
                                                                     {1,0,-1},
                                                                     {1,0,-1}},

                                                                    {{0,1,0},  /* 65 */
                                                                     {1,0,-1},
                                                                     {0,-1,0}},

                                                                    {{0,0,1},  /* 45 */
                                                                     {0,1,0},
                                                                     {1,0,0}},

                                                                    {{0,-1,0},  /* 25 */
                                                                     {-1,0,1},
                                                                     {0,1,0}},

                                                                    {{-1,-1,-1},  /* 0 */
                                                                     {0,0,0},
                                                                     {1,1,1}}
                                                                   };
#endif

#ifdef V2S_FILTERS_3
float v2s_filters[V2S_FILTER_CHANNELS][V2S_FILTER_ROWS][V2S_FILTER_COLS] = {
                                                                    {{1,1,1}, /* 180 horizontal */
                                                                     {0,0,0},
                                                                     {0,0,0}},

                                                                    {{0,0,0},  /* 155 */
                                                                     {1,0,0},
                                                                     {0,1,0}},

                                                                    {{1,0,0},  /* 135 */
                                                                     {0,1,0},
                                                                     {0,0,1}},

                                                                    {{0,1,0},  /* 115 */
                                                                     {0,0,1},
                                                                     {0,0,0}},

                                                                    {{1,0,0},  /* 90 */
                                                                     {1,0,0},
                                                                     {1,0,0}},

                                                                    {{0,1,0},  /* 65 */
                                                                     {1,0,0},
                                                                     {0,0,0}},

                                                                    {{0,0,1},  /* 45 */
                                                                     {0,1,0},
                                                                     {1,0,0}},

                                                                    {{0,0,0},  /* 25 */
                                                                     {0,0,1},
                                                                     {0,1,0}},

                                                                    {{0,0,0},  /* 0 */
                                                                     {0,0,0},
                                                                     {1,1,1}}
                                                                   };
#else
float v2s_filters[V2S_FILTER_CHANNELS][V2S_FILTER_ROWS][V2S_FILTER_COLS] = {
                                                                    {{1,1,1,1,1}, /* 180 horizontal */
                                                                     {0,0,0,0,0},
                                                                     {0,0,0,0,0},
                                                                     {0,0,0,0,0},
                                                                     {0,0,0,0,0}},

                                                                    {{0,0,0,0,0},  /* 155 */
                                                                     {1,0,0,0,0},
                                                                     {0,1,0,0,0},
                                                                     {0,0,1,0,0},
                                                                     {0,0,0,1,0}},

                                                                    {{1,0,0,0,0},  /* 135 */
                                                                     {0,1,0,0,0},  
                                                                     {0,0,1,0,0},  
                                                                     {0,0,0,1,0},  
                                                                     {0,0,0,0,1}},

                                                                    {{0,1,0,0,0},  /* 115 */
                                                                     {0,0,1,0,0},
                                                                     {0,0,0,1,0},
                                                                     {0,0,0,0,1},
                                                                     {0,0,0,0,0}},

                                                                    {{1,0,0,0,0},  /* 90 */
                                                                     {1,0,0,0,0},
                                                                     {1,0,0,0,0},
                                                                     {1,0,0,0,0},
                                                                     {1,0,0,0,0}},

                                                                    {{0,0,0,1,0},  /* 65 */
                                                                     {0,0,1,0,0},
                                                                     {0,1,0,0,0},
                                                                     {1,0,0,0,0},
                                                                     {0,0,0,0,0}},

                                                                    {{0,0,0,0,1},  /* 45 */
                                                                     {0,0,0,1,0},
                                                                     {0,0,1,0,0},
                                                                     {0,1,0,0,0},
                                                                     {1,0,0,0,0}},

                                                                    {{0,0,0,0,0},  /* 25 */
                                                                     {0,0,0,0,1},
                                                                     {0,0,0,1,0},
                                                                     {0,0,1,0,0},
                                                                     {0,1,0,0,0}},

                                                                    {{0,0,0,0,0},  /* 0 */
                                                                     {0,0,0,0,0},
                                                                     {0,0,0,0,0},
                                                                     {0,0,0,0,0},
                                                                     {1,1,1,1,1}}
								   }; 
#endif

/* functions called in this file by main */

int load_image(unsigned char image[][IMAGE_NUM_PIXELS], char *file_name);
static void construct_filter_exp(float filter[][CS_NUM_COLS]);
static int populate_cones(unsigned char cones[][CONES_NUM_COLS], unsigned char image[][IMAGE_NUM_PIXELS]);
static int compute_rgc_output(float rgc_cells[][RGC_NUM_COLS], unsigned char cones[][CONES_NUM_COLS], float filter[][CS_NUM_COLS]);
static int compute_v1s_exp(float v1s[][V1S_NUM_ROWS][V1S_NUM_COLS], float rgc[][RGC_NUM_COLS], float filters[][V1_FILTER_ROWS][V1_FILTER_COLS]);
static int compute_v1c(float v1c[][V1C_NUM_ROWS][V1C_NUM_COLS], float v1s[][V1S_NUM_ROWS][V1S_NUM_COLS]);
static int compute_v2s_full(void);
static int compute_v2c(void);
static int load_v4s_filter(void);
static void print_v4s_filter(void);
static int learn_v4s_filter(void);
static void store_v4s_filter(void);
static int compute_v4s_full(void);
static void compute_hash(void);
static void print_hash(void);
static int process_hash_in_it(void);
static void load_world_state(void);
static void store_world_state(void);

/******************** functions not called by main *********************************/

static int is_surround(int i, int k);
static int is_center(int i, int k);
static float activation(float input);
static int is_v2c_spatial_correlation_present(int feature1, int feature2, int i, int j);
static float convolve_rgc(unsigned char cones[][CONES_NUM_COLS], float filter[][CS_NUM_COLS], int c1, int c2);
static float get_nth_value(int n, int ch, int row_start, int col_start);
static float convolve_v2s_new(int f1, int f2, int c1, int c2);
static float convolve_v2s_new_array(int channel_list[], int num_channels, int c1, int c2);
static float convolve_v1(float rgc[][RGC_NUM_COLS], float filter[][V1_FILTER_COLS], int c1, int c2);
static float pool_v1c(float v1s[V1S_NUM_ROWS][V1S_NUM_COLS], int c1, int c2);
static float pool_v2c(float v2s[V2S_NUM_ROWS][V2S_NUM_COLS], int c1, int c2);
static int compute_v2s_new(int f1, int f2, int ch);
static int compute_v2s_new_array(int channel_list[], int num_channels, int ch);
static int init_v4s_filter_random(void);
static void get_feature_combinations(int ch, int *feature1, int *feature2);
static int adjust_weights(int ch, int i, int j);
static float convolve_v4s(int feature1, int feature2, int c1, int c2, int filter_channel);
static int compute_v4s(int f1, int f2, int ch);
static int associate_hash_with_world(void);
static int find_the_digits(void);

/* main */

int main(int argc, char *argv[])
{
	int rc;
	float filter[CS_NUM_ROWS][CS_NUM_COLS];

	if(argc > 1) {
		printf("Using user supplied image filename %s\n", argv[1]);
		rc = load_image(image, argv[1]);
		if(argc > 2) {
			label = atoi(argv[2]);
			train_mode = 1;
		} else {
			train_mode = 0;
		}
	} else {
		printf("Using default file for image\n");
		rc = load_image(image, IMAGE_FILE_NAME);
		printf("Provide an input image file name\n");
		return 0;
	}

	if(rc == -1) {
		return -1;
	}

	load_world_state();

	if(train_mode) {
		print_world_state();
	}

	printf("Below is the image %d %d\n", IMAGE_NUM_PIXELS, IMAGE_NUM_PIXELS);
	print_image(image);

	printf("cones rows and  columns: %d %d\n", CONES_NUM_ROWS, CONES_NUM_COLS);
	printf("output retina rows and  columns: %d %d\n\n", RGC_NUM_ROWS, RGC_NUM_COLS);

	populate_cones(cones, image);

#if 0 
	printf("Below are the cones\n");
	print_cones(cones);
#endif

        // load_abcd();
 
	construct_filter_exp(filter); /* This is for experimenting */
	//printf("Below is the filter\n");
	//print_filter(filter);
	/* rgc output from the photoreceptor layer */
	//printf("Computing RGC Output\n");

	compute_rgc_output(rgc_cells, cones, filter);
	//printf("Below is the RGC Output %d x %d\n\n", RGC_NUM_ROWS, RGC_NUM_COLS);
	//print_rgc(rgc_cells);

	/* get the v1s output for all of the filters */
	compute_v1s_exp(v1s, rgc_cells, v1_filters);
	//printf("Below is the V1S output %d %d\n", V1S_NUM_ROWS, V1S_NUM_COLS);
	//print_v1s(v1s);

	compute_v1c(v1c, v1s); //this just does the pooling
	//printf("Below is the V1C output %d x %d:\n\n", V1C_NUM_ROWS, V1C_NUM_COLS);
	//print_v1c(v1c);

	//printf("Computing v2s\n");
	compute_v2s_full(); //compute v2s for all channels, each channel corresponds to a feature(top curve, etc)
	//printf("Below is V2S: %d %d\n\n", V2S_NUM_ROWS, V2S_NUM_COLS);
	//print_v2s(v2s);

	compute_v2c();
	//printf("Below is V2C: %d %d\n\n", V2C_NUM_ROWS, V2C_NUM_COLS);
	//print_v2c(v2c);

	load_v4s_filter();
	//printf("Below is v4s filter after loading\n");
	//print_v4s_filter();

	//we learn on all inputs by default. do we learn on both training and when detecting?
	//but when we load we will load the values learnt previously
	printf("Doing the learning\n");
	learn_v4s_filter();
	//printf("Below is v4s filter after learning\n");
	//print_v4s_filter();

	//store the learned v4s filter values into a file
	store_v4s_filter();

	compute_v4s_full();
	//printf("Below is V4S:\n\n");
	//print_v4s(v4s);

	compute_hash();
	//print_hash();

        // pass the hash to proto, and measure the response
        process_hash_in_it();

	if(train_mode) {
		store_world_state();
	}

	// the proto should be a running thing, the incoming hash makes the proto recognize something
}

void print_world_state(void)
{
	int i;
	int j;
	float hash;

	printf("TOTAL_ELEMENTS: %d\n", TOTAL_ELEMENTS);
	printf("printing non-zero elements of hash\n");

	for(i=0; i<10; i++) {
		printf("Digit: %d\n", i);
		for(j=0; j<TOTAL_ELEMENTS; j++) {
			hash = world_state[i].hash[j];
			if(hash != 0) {
				printf("index: %d hash: %f \n", j, hash);
			}
		}
	}
}

void load_world_state(void)
{
	/*
	 * World state file format:
	 * A single hash: TOTAL_ELEMENTS float value
	 * 10 hashes for 10 digits
	 */

	int i;
	FILE *fptr;

	for(i=0; i<10; i++) {
		memset(world_state[i].hash, 0, sizeof(float) * TOTAL_ELEMENTS);
	}

	fptr = fopen(HASH_FILE_NAME, "rb");

	if(!fptr) {
		printf("hash file not found\n");
		return;
	}

	fread(world_state, sizeof(world_state_t), 10, fptr);

	fclose(fptr);
}

void store_world_state(void)
{
	/*
	 * World state file format:
	 * A single hash: TOTAL_ELEMENTS float value
	 * 10 hashes for 10 digits
	 */

	int i;

	FILE *fptr;

	fptr = fopen(HASH_FILE_NAME, "wb");

	if(!fptr) {
		printf("hash file not found\n");
		return;
	}

	fwrite(world_state, sizeof(world_state_t), 10, fptr);

	fclose(fptr);
}

static void copy_hash(float hash[TOTAL_ELEMENTS])
{
	int i;

	for(i=0; i<TOTAL_ELEMENTS; i++) {
		if(hashcode[i] != 0) {
			hash[i] = INITIAL_HASH_VALUE;
		}
	}
}

static int associate_hash_with_world(void)
{
	/*
	 * Check if you already learnt this particular label,
	 * If you have, then the new data for the label 
	 * should reinforce the existing association between the data and the label
	 *
	 * If you haven't seen this label before, then just store this hash asis
	 * for this label (world state)
	 */

	int i;

	if(world_state[label].learned == 0) {
		copy_hash(world_state[label].hash);
		world_state[label].learned = 1;
	} else {
		/* 
		 * reinforce the hash : We compare the incoming hash to the existing one
		 * If the element is ON in both the incoming hash and existing hash
		 * we increase its value
		 * If the element is present in existing hash but not in incoming hash then we decrease it a little
		 */

		for(i=0; i<TOTAL_ELEMENTS; i++) {
			if(world_state[label].hash[i] != 0) {

				if(hashcode[i]) {
					world_state[label].hash[i] += 0.3;
				} else {
					world_state[label].hash[i] -= 0.1;
				}
			} else {
				world_state[label].hash[i] = INITIAL_HASH_VALUE;
			}
		}
	}

	return 0;
}

float mod(float d)
{
	if(d < 0) {
		return d*(-1.0);
	}

	return d;
}

float find_distance(int digit)
{
	int i;
	int d = 0.0;

	/*
	 * Find the distance between the incoming hash and the hash for the digit in
	 * the world_state
	 */

	for(i=0; i<TOTAL_ELEMENTS; i++) {
		d += mod((float)world_state[digit].hash[i] - hashcode[i]);
	}

	return d;

}


static int find_the_digits(void)
{
	/*
	 * The world already has knowledge of the hash of the labels (world states)
	 * You have to figure out which of the world states this hash matches to closely
	 */

	/*
	 * You can calculate a hash distance between the computed hash and the world hashes and see
	 * which one produces the lowest score
	 */

	int i;
	float scores[10];
	float score = 0.0;
	int digit = 0;

	for(i=0; i<10; i++) {
		scores[i] = find_distance(i);
		printf("score for %d: %f\n", i, scores[i]);
		if(i == 0) {
			score = scores[i];
		} else {
			if(score > scores[i]) {
				score = scores[i];
				digit = i;
			}
		}
	}

	return digit;
}

static int process_hash_in_it(void)
{
	int d;

	if(train_mode) {
		associate_hash_with_world();
	} else {
		d = find_the_digits();
		printf("given digit is %d\n", d);
	}

	return 0;
}

int load_image(unsigned char image[][IMAGE_NUM_PIXELS], char *file_name)
{
        int i,j;
        FILE *filp;

        if(file_name == NULL)
                return -1;

        filp = fopen(file_name, "r");

        if(filp == NULL) {
                printf("image file not present\n");
                return -1;
        }

        for(i=0; i<IMAGE_NUM_PIXELS; i++) {
                for(j=0; j<IMAGE_NUM_PIXELS; j++) {
                        fscanf(filp, "%u", &image[i][j]);
                }
        }

        fclose(filp);
        return 0;
}

/* We focus the image onto photoreceptors */
/* We expand the image pixels into photoreceptors column-wise */

static int populate_cones(unsigned char cones[][CONES_NUM_COLS], unsigned char image[][IMAGE_NUM_PIXELS])
{
	unsigned int i=0, k=0, m=0, n=0, col;
	unsigned int pixel_value, cone_value;

	memset(cones, 0, CONES_NUM_ROWS * CONES_NUM_COLS * sizeof(unsigned char));

	for(i=0, m=0; i<CONES_NUM_ROWS; i++, m++) {
		//printf("populate_cones: populating row: %d\n", i);
		for(k=0, n=0; k<CONES_NUM_COLS; k+=CONES_PER_PIXEL, n++) {
			pixel_value = (unsigned int)(image[m][n]); //it will be anywhere between 0 and 255
			cone_value = pixel_value / CONES_PER_PIXEL;
			//printf("populating col %u with value %u\n", k, pixel_value);
			for(col=k; col<k+CONES_PER_PIXEL; col++) { //CONES_PER_PIXEL tells you how you want to split the pixel value
				cones[i][col] = cone_value;
			}
		}
	}

	return 0;
}

/* This filter is for experimenting purpose. This filter will do nothing, pass cones asis to rgc */

static void construct_filter_exp(float filter[][CS_NUM_COLS])
{
	int i, k;

	for(i=0; i<CS_NUM_ROWS; i++) {
		for(k=0; k<CS_NUM_COLS; k++) {
			filter[i][k] = 1;
		}
	}
}

static int compute_rgc_output(float rgc_cells[][RGC_NUM_COLS], unsigned char cones[][CONES_NUM_COLS], float filter[][CS_NUM_COLS])
{
	int i, k;
	float output;
	
	/* first initialize retinal cells with 0, so that the cells that do not get input from cones
         * due to size restrictions will be initialized with a value
	 */

	memset(rgc_cells, 0, RGC_NUM_ROWS * RGC_NUM_COLS * sizeof(float));

	for(i=0; i<RGC_NUM_ROWS; i++) {
		for(k=0; k<RGC_NUM_COLS; k++) {
			output = convolve_rgc(cones, filter, i, k);
			rgc_cells[i][k] = activation(output);
		}
	}

	printf("compute_rgc_output done\n");
	return 0;
}

/* Experiment version: This uses only one filter for all positions of rgc_cells
 * But then it sequentially does this with multiple filters and thus producing multiple results
 */

static int compute_v1s_exp(float v1s[][V1S_NUM_ROWS][V1S_NUM_COLS], float rgc[][RGC_NUM_COLS], float filters[][V1_FILTER_ROWS][V1_FILTER_COLS])
{
	int i, k, c;
	
	/* set output to all zero */
	memset(v1s, 0, V1S_NUM_CHANNELS * V1S_NUM_ROWS * V1S_NUM_COLS * sizeof(float));

	for(c=0; c<V1S_NUM_CHANNELS; c++) {
		for(i=0; i<V1S_NUM_ROWS; i++) {
			for(k=0; k<V1S_NUM_COLS; k++) {
				//rgc is 2d, filters[c] is 2d, they convolve and give scalar
				v1s[c][i][k] = convolve_v1(rgc, filters[c], i, k);
			}
		}
	}

	printf("compute_v1s_exp done\n");
	return 0;
}

/* v1c is v1 complex1 layer. Complex layers just do pooling */
static int compute_v1c(float v1c[][V1C_NUM_ROWS][V1C_NUM_COLS], float v1s[][V1S_NUM_ROWS][V1S_NUM_COLS])
{
	int i, k, c;
	//we should do pooling here
	memset(v1c, 0, V1C_NUM_CHANNELS * V1C_NUM_ROWS * V1C_NUM_COLS * sizeof(float));

	for(c=0; c<V1C_NUM_CHANNELS; c++) {
		for(i=0; i<V1C_NUM_ROWS; i++) {
			for(k=0; k<V1C_NUM_COLS; k++) {
				v1c[c][i][k] = pool_v1c(v1s[c], i, k);
			}
		}
	}

	printf("compute_v1c done\n");
	return 0;
}

static int compute_v2s_full(void)
{
	int channel_list[10] = {};
	int n_channels;

	memset(v2s, 0, V2S_NUM_CHANNELS * V2S_NUM_ROWS * V2S_NUM_COLS * sizeof(float));

	/*
	 * Degrees             : 180  155  135  115  90  65  45  25  0 
	 * Indices in filter   : 0    1    2    3    4   5   6   7   8
	 *
	 * The last parameter to compute_v2s_new is the channel in v2s that we want to fill
	 * the first 2 parameters are the channels in v1c that we are mating
	 * So the channel(last parameter) corresponds to the feature in the V1s2
	 */

	compute_v2s_new(3, 7, 0);//right curve
	compute_v2s_new(5, 1, 1);//left curve
	compute_v2s_new(5, 3, 2);//top curve
	compute_v2s_new(1, 7, 3);//bottom curve

        channel_list[0] = 5;
        channel_list[1] = 3;
        channel_list[2] = 1;
        channel_list[3] = 7;
	n_channels = 4;

	compute_v2s_new_array(channel_list, n_channels, 4);//circle, here we are mating a list of channels

	return 0;
}

static int compute_v2c(void)
{
	int i, k, c;
	//we should do pooling here
	memset(v2c, 0, V2C_NUM_CHANNELS * V2C_NUM_ROWS * V2C_NUM_COLS * sizeof(float));

	for(c=0; c<V2C_NUM_CHANNELS; c++) {
		for(i=0; i<V2C_NUM_ROWS; i++) {
			for(k=0; k<V2C_NUM_COLS; k++) {
				v2c[c][i][k] = pool_v2c(v2s[c], i, k);
			}
		}
	}

	printf("compute_v2c done\n");
	return 0;
}

static int load_v4s_filter(void)
{
	/*
	 * This function will load the learned filter values for v4s from file into the filters variable
	 * For now it just uses random values
	 */

	FILE *fptr = NULL;

	fptr = fopen(V4S_FILTER_FILE_NAME, "r");

	if(fptr) {
		printf("\n load_v4s_filter: Filter file found, using it for filter values\n");
		fread(v4s_filters, sizeof(v4s_filters), 1, fptr);
		fclose(fptr);
	} else {
		printf("\n load_v4s_filter: filter file not found, using the random values for filter\n");
		init_v4s_filter_random();
	}

	return 0;
}

static int learn_v4s_filter(void)
{
	int ch, i, j;

	/*
	 * Input:
	 *  1. The v4s_filter variable
	 *  2. The input image
         *  3. The V2C features
	 *  4. The label
	 * 
	 * Processing:
	 * 1. Find the spatial correlation in the image
	 *
	 * Output:
	 * 1. The v4s_filter variable with changed values
	 */

	/*
	 * We need to make some values 0 and others 1
	 * Which values should be zero? The values where the features don't interact.
	 * So easier question: Which  values should be one?
	 * The better question is which couple or triplet or quartet or five values should be one?
	 * The answer is the group of values whose features appear combined in the given input
	 * How do we adjust the values in the backgroud of the previous inputs?
	 * We don't want the latest input to ovethrow everyother input
	 * So we need to make a decision. 
	 */

	for(ch=0; ch<V4S_FILTER_NUM_CHANNELS; ch++) {
		for(i=0; i<V4S_FILTER_ROWS; i++) {
			for(j=0; j<V4S_FILTER_COLS; j++) {
			       adjust_weights( ch, i, j );
				
			}
		}
	}

	return 0;  
}

static void store_v4s_filter(void)
{
	FILE *fptr;
	char v4s_filter_file_name[128] = {'\0'};
	time_t timeval = 0;

	printf("\nStoring the v4s filter\n");

	time(&timeval);

	sprintf(v4s_filter_file_name, "./%s/%s_%d", BKUP_FILES_DIR_NAME, V4S_FILTER_FILE_NAME, (int)timeval);
	printf("backup file name is %s\n", v4s_filter_file_name);

	//make a copy 
	rename(V4S_FILTER_FILE_NAME, v4s_filter_file_name);

	// now write the new filter values to the file
	fptr = fopen(V4S_FILTER_FILE_NAME, "w");

	if(fptr == NULL) {
		printf("can not open file %s", V4S_FILTER_FILE_NAME);
		return;
	}

	fwrite(v4s_filters, sizeof(v4s_filters), 1, fptr);
	fclose(fptr);
}

/*
 * We need only 5 filters because the number of features from our input V2C is 5.
 * When we mate the v2c features 0 and 1, we use the V4 filters 0 and 1, and so on.
 * So even though we compute 10 features for V4S here, we only need 5 filters because 
 * our input has only 5 features.
 * The filters basically tell us how to mate.
 * They tell us which two numbers to multiply from the two input features.
 * The filters are only 1 or 0 (as of now)
 * What they tell us, is that we need to mate corresponding features in the two input
 * We need to select an element from input1 and an element from input2.
 * We use filter1 to select an element from input1.
 * We use filter2 to select an element from input2.
 */

static int compute_v4s_full(void)
{
	memset(v4s, 0, V4S_NUM_CHANNELS * V4S_NUM_ROWS * V4S_NUM_COLS * sizeof(float));

	/* we need 10 filters */

	compute_v4s(0, 1, 0); //mate the first 2 features in v2c and put it in index 0, use the filter 0 for that.
                               // the last argument serves as both the output v4s index  and the filter channel index
	compute_v4s(0, 2, 1); //mate the first and third features in v2c and put it in index 1
	compute_v4s(0, 3, 2); //mate the first and fourth features in v2c and put it in index 2
	compute_v4s(0, 4, 3); //mate the first and fifth features in v2c and put it in index 3
	compute_v4s(1, 2, 4); //mate the second and third features in v2c and put it in index 4
	compute_v4s(1, 3, 5); //mate the second and fourth  features in v2c and put it in index 5
	compute_v4s(1, 4, 6); //mate the second and fifth features in v2c and put it in index 6
	compute_v4s(2, 3, 7); //mate the third and fourth features in v2c and put it in index 7
	compute_v4s(2, 4, 8); //mate the third and fifth features in v2c and put it in index 8
	compute_v4s(3, 4, 9); //mate the fourth and fifth features in v2c and put it in index 9

	return 0;
}

static void compute_hash(void)
{
	int i, j, k, index = 0;
	int value;

	/*
	 * First fill out the hash corresponding to the input
	 */

	for(i=0; i<IMAGE_NUM_PIXELS; i++) {
		for(j=0; j<IMAGE_NUM_PIXELS; j++) {
			value = image[i][j] > IMAGE_THRESHOLD;	
			hashcode[index++] = value;
		}
	}

	/* 
	 * Now go through each layers and fill out the hash code
	 */

	for(i=0; i<V1S_NUM_CHANNELS; i++) {
		for(j=0; j<V1S_NUM_ROWS; j++) {
			for(k=0; k<V1S_NUM_COLS; k++) {
				value = v1s[i][j][k] > V1S_THRESHOLD;
				hashcode[index++] = value;
			}
		}
	}

	for(i=0; i<V1C_NUM_CHANNELS; i++) {
		for(j=0; j<V1C_NUM_ROWS; j++) {
			for(k=0; k<V1C_NUM_COLS; k++) {
				value = v1c[i][j][k] > V1C_THRESHOLD;
				hashcode[index++] = value;
			}
		}
	}

	for(i=0; i<V2S_NUM_CHANNELS; i++) {
		for(j=0; j<V2S_NUM_ROWS; j++) {
			for(k=0; k<V2S_NUM_COLS; k++) {
				value = v2s[i][j][k] > V2S_THRESHOLD;
				hashcode[index++] = value;
			}
		}
	}

	for(i=0; i<V2C_NUM_CHANNELS; i++) {
		for(j=0; j<V2C_NUM_ROWS; j++) {
			for(k=0; k<V2C_NUM_COLS; k++) {
				value = v2c[i][j][k] > V2C_THRESHOLD;
				hashcode[index++] = value;
			}
		}
	}

	for(i=0; i<V4S_NUM_CHANNELS; i++) {
		for(j=0; j<V4S_NUM_ROWS; j++) {
			for(k=0; k<V4S_NUM_COLS; k++) {
				value = v4s[i][j][k] > V4S_THRESHOLD;
				hashcode[index++] = value;
			}
		}
	}
}

static void print_hash(void)
{
	int i, pindex = 0;

	for(i=0; i<TOTAL_ELEMENTS; i++) {
		if (hashcode[i] != 0) {
			num_positions++;
			positions[pindex++] = i;
		}
	}

	printf("Number of one positions are %d total: %d\n", num_positions, TOTAL_ELEMENTS);
	printf("Below are the positions\n\n");

	for(i=0; i<num_positions; i++) {
		if(i % 20 == 0) {
			printf("\n");
		}
		printf("  %d", positions[i]);
	}

	printf("\n");
}

/****************** Here functions not called by main ******************************/

static int is_surround(int i, int k)
{
	if(i == 0 || i == CS_NUM_ROWS -1) {
		return 1;
	} else {
		if(k == 0 || k == CS_NUM_COLS -1) {
			return 1;
		}
	}
	return 0;
}

static int is_center(int i, int k)
{
	return !(is_surround(i, k));
}

static float activation(float input)
{
	return relu(input);
}

/*
 * find out if there is spatial correlation between
 * the given features in V2C
 */
static int is_v2c_spatial_correlation_present(int feature1, int feature2, int i, int j)
{
        //you can try 0.1, 0.2 for the threshold
        //the position_importance values are in the range of 0.08 to 1
        float THRESHOLD = 0.0101;

        float position_importance, position_correlation;
        int row, col;
        float value1, value2;
        float output = 0.0;

        /*
         * The V2C values are very high integers like 2000 or so. Take that into account
         * when setting threshold.
         */

        //printf("%s: feature1: %d feature2: %d i: %d j: %d\n", __FUNCTION__, feature1, feature2, i, j);

        /*
         * What is the dimension of v2c? 5 channels.
         * The feature1 and feature2 can be anywhere between 0 - 4, representing any of the 5 channels in v2c,
         * how do we find the correlation between the given 2 features?
         * We can see if the two channels have 1s in the corresponsing positions?
         * We can multiply the positions and see if they exceed a threshold?
         */

#if 0
        printf("calling matrix mult\n");
        scalar_result = mat_mult(v2c[feature1], v2c[feature2]);
#endif

        for(row=0; row<V2C_NUM_ROWS; row++) {
                for(col=0; col<V2C_NUM_COLS; col++) {
                        value1 = v2c[feature1][row][col];
                        value2 = v2c[feature2][row][col];

                        output += value1 * value2;
                }
        }

        /*
         * Now we need to find out how much the postions i, and j (the indices in the filter)
         * contributed to the oveall scalalr result of feature multiplication.
         *
         * Here we make use of our knowledge that V2C rows, cols is 5 * 5 which is same
         * as the filter size (v4s filter). But it doesn't have to be same. It just so happens to be same here
         * We also have to take care of sliding.
         */

        position_correlation = v2c[feature1][i][j] * v2c[feature2][i][j];
        position_importance = position_correlation / output; //how much position correlation contributed to the overall features multiplication

        /* Todo: maybe we can pass position_importance to an activation function that returns between 0 and 1 */

        //printf("correlation: %f output: %f position_importance: %f \n", position_correlation, output, position_importance);

        return (position_importance > THRESHOLD);
}

/*
 * Question: do we simply convolve or do we do something else?
 * The surrond in our filters is horizontal cells. They act like sign switchers
 * But what do those cells do when they encounter 0 (no light)? Do they switch it to some
 * small positive? or do they just ignore? 
 * I mean when there is a black input(0 ie no light) do they contribute to the firing of the cs unit?
 * If so horizontal cells are not just sign switchers, they also convert no-light into a contribution to the center.
 * How to settle this? One test is Let's shine a small light (a bar) on the edge of the center and measure the firing
 * Now shine a light same as before at the edge of a center but disconnect input from the surrond cell, what happens now?
 * Is the firing = ap from light or firing = ap from light + ap from black?
 * If it's ap from light + ap from black, then we may need to do something even when we encounter 0 at the surrond
 * Maybe when we encounter zero at the surrond, make that zero -0.1 or something
 */

/* This is just one step of convolution */

static float convolve_rgc(unsigned char cones[][CONES_NUM_COLS], float filter[][CS_NUM_COLS], int c1, int c2)
{
	float output = 0.0, cone_value = 0.0;
	int i, k, i2, k2, center_bright = 0;

	/* i2, k2 are  indices into the cones, i and k are indices to the filter */

	for(i=0, i2 = (c1*STRIDE); i<CS_NUM_ROWS; i++, i2++) { //i2 and k2 calc needs changing if padding changes from 0
		for(k=0, k2 = (c2*STRIDE); k<CS_NUM_COLS; k++, k2++) {
			if(i2 >= CONES_NUM_ROWS || i2 < 0 || k2 >= CONES_NUM_COLS || k2 < 0) {
				printf("i2 or k2 value not right i2: %d k2: %d c1: %d c2: %d\n", i2, k2, c1, c2);
				cone_value = 0.0;
			}
			else {
				cone_value = cones[i2][k2];
			}
			if(is_center(i, k) && cone_value > 0) {
				center_bright = 1;
			}
			if(is_surround(i, k) && cone_value == 0) {
				/* So our surround cell caught a zero, let it contribute a little to the center */
				/* But we let itt contribute only if one of center is nonzero */
				if(center_bright == 1)
					output += 0.1; //it used to be 0.1
			} else {
				output += (float)cone_value * filter[i][k]; //this can be surround or center
			}
		}
	}
	return output;
}

/*
 * This should first get the indices in the filter of nth 1,
 * And then return the value in that index in v1c matrix for that particular convolution
 */

static float get_nth_value(int n, int ch, int row_start, int col_start)
{
	int i, k, c = 0, i2, k2;

	for(i=0; i<V2S_FILTER_ROWS; i++) {
		for(k=0; k<V2S_FILTER_COLS; k++) {
			if(v2s_filters[ch][i][k] > 0.0) { //channel is constant here
				if(c == n) {
					i2 = row_start + i;
					k2 = col_start + k;
					//printf("get_nth_value: returning %f ch:%d n: %d i2:%d k2:%d\n", v1c[ch][i2][k2], ch, n, i2, k2);
					return v1c[ch][i2][k2];
				}
				c++;
			}
		}
	}
	//printf("get_nth_value: returning default 0 ch:%d n: %d \n", ch, n);
	return 0.0;
}

/*
 * Here we are mating 2 input features f1 and f2.
 * The input are from v1c. This function is called each time 
 * the sliding happens over the two features.
 * So the f1 and f2 may remain constant as we slide over and change the c1 and c2
 * This function returns the result of one such slide.
 * For returning a scalar for one such slide what we need to do is, we need to multiply
 * elements from f1 and f2 of v1c. How do we find the elements? That's where the filters will help.
 * Let's say the filters each have 3 ones in 3 different places. What we will do is we will get the 
 * first corresponding pair of elements from the 2 inputs and multiply these 2 numbers and store it in result
 * Then we will do the same for second and third couple of elements.
 * Then we will sum the 3 sub-results and return it as final result
 */

static float convolve_v2s_new(int f1, int f2, int c1, int c2)
{
	float output = 0.0, value1 = 0.0, value2 = 0.0;
	int i, k, n = 0;
	int row_start, col_start, row_end, col_end;

	row_start = (c1*V2S_STRIDE) - V2S_PADDING;
	col_start = (c2*V2S_STRIDE) - V2S_PADDING;

	row_end = row_start + V2S_FILTER_ROWS;
	col_end = col_start + V2S_FILTER_COLS;
	
	/* this looping is basically loopping over the filter size */
	for(i=row_start; i<row_end; i++) {
		for(k=col_start; k<col_end; k++) {
			/* this boundary check is helpful when we have padding */
			if(i < 0 || k < 0 || i >= V1C_NUM_ROWS || k >= V1C_NUM_COLS) {
				printf("%s: i or k value invalid: i:%d k:%d\n", __FUNCTION__, i, k);
				value1 = 0.0;
				value2 = 0.0;
			}
			else {
				value1 = get_nth_value(n, f1, row_start, col_start);
				value2 = get_nth_value(n, f2, row_start, col_start);
				n++;
			}
			output += value1 * value2;
			//printf("-------------------\n");
		}
	}
	return output;
}

/* The array version of of convolve_v2s_new. operates on list of channels instead of two */

static float convolve_v2s_new_array(int channel_list[], int num_channels, int c1, int c2)
{
	float output = 0.0, value = 0.0, product = 1.0;
	int i, k, n = 0, c;
	int row_start, col_start, row_end, col_end;

	row_start = (c1*V2S_STRIDE) - V2S_PADDING;
	col_start = (c2*V2S_STRIDE) - V2S_PADDING;

	row_end = row_start + V2S_FILTER_ROWS;
	col_end = col_start + V2S_FILTER_COLS;
	
	for(i=row_start; i<row_end; i++) {
		for(k=col_start; k<col_end; k++) {
			/* this boundary check is useful when we have padding */
			if(i < 0 || k < 0 || i >= V1C_NUM_ROWS || k >= V1C_NUM_COLS) {
				printf("%s: i or k value invalid: i:%d k:%d\n", __FUNCTION__, i, k);
				product = 0;
			}
			else {
				product = 1.0;
				for(c=0; c<num_channels; c++) { //multiply the values in all the channels
					value = get_nth_value(n, channel_list[c], row_start, col_start);
					product = product * value;
				}
				n++;
			}
			output += product;
		}
	}
	return output;
}

static float convolve_v1(float rgc[][RGC_NUM_COLS], float filter[][V1_FILTER_COLS], int c1, int c2)
{
	float output = 0.0, value = 0.0;
	int i, k, i2, k2;

	/* i2, k2 are  indices to the rgc, i and k are indices to the filter */

	i2 = (c1*V1_STRIDE) - V1_PADDING;

	for(i=0; i<V1_FILTER_ROWS; i++, i2++) {
		k2 = (c2*V1_STRIDE) - V1_PADDING;
		for(k=0; k<V1_FILTER_COLS; k++, k2++) {

			if(i2 < 0 || k2 < 0 || i2 >= RGC_NUM_ROWS || k2 >= RGC_NUM_COLS) {
				printf("convolve_v1: i2 or k2 value invalid: i2:%d k2:%d\n", i2, k2);
				value = 0.0;
			}
			else {
				value = rgc[i2][k2];
			}
			/*
			 * This convolution essentially sees if there is a line in the input. 
			 * If there is a 1 in our filter that means we expect an input in the corresponding position
			 * in the rgc for a complete line to form. 
			 * That's why if there is a 1 in filter and 0 in corresponding rgc, we detect there is no line.
			 * So we return 0. This step is important, it discerns features well.
			 */ 
			if(filter[i][k] == 1 && value <= 0.0) {
				return 0.0;
			}

			output += value * filter[i][k];
		}
	}
	return output;
}

/* This is a max pooling layer */
static float pool_v1c(float v1s[V1S_NUM_ROWS][V1S_NUM_COLS], int c1, int c2)
{
	int i, k, i_start, k_start; /* i and k are indices to the v1s */
	float max;

	i_start = c1 * V1C_POOL_ROWS;
	k_start = c2 * V1C_POOL_COLS;

	max = v1s[i_start][k_start];

	for(i=i_start; i<i_start+V1C_POOL_ROWS && i<V1S_NUM_ROWS; i++) {
		for(k=k_start; k<k_start+V1C_POOL_COLS && k<V1S_NUM_COLS; k++) {
			if(v1s[i][k] > max) {
				max = v1s[i][k];
			}
		}
	}
	return max;
}

static float pool_v2c(float v2s[V2S_NUM_ROWS][V2S_NUM_COLS], int c1, int c2)
{
	int i, k, i_start, k_start; /* i and k are indices to the v2s */
	float max;

	i_start = c1 * V2C_POOL_ROWS;
	k_start = c2 * V2C_POOL_COLS;

	max = v2s[i_start][k_start];

	for(i=i_start; i<i_start+V2C_POOL_ROWS && i<V2S_NUM_ROWS; i++) {
		for(k=k_start; k<k_start+V2C_POOL_COLS && k<V2S_NUM_COLS; k++) {
			if(v2s[i][k] > max) {
				max = v2s[i][k];
			}
		}
	}
	return max;
}

/*
 * v2s is the second simple layer.It selects (ANDS) features from the v1c
 * f1, f2 represents the channels in the v1c that we want to AND
 * ch represents the channel in v2s that we are calculating
 */

static int compute_v2s_new(int f1, int f2, int ch)
{
	int i, k;

	for(i=0; i<V2S_NUM_ROWS; i++) {
		for(k=0; k<V2S_NUM_COLS; k++) {
			v2s[ch][i][k] = convolve_v2s_new(f1,f2, i, k);
			//printf("v2s[%d][%d][%d]: %0.2f\n", ch, i, k, v2s[ch][i][k]);
		}
	}

	return 0;
}

/* same as compute_v2s_new. But instead of operating on 2 channels,
 * it operates on list of channels
 */

static int compute_v2s_new_array(int channel_list[], int num_channels, int ch)
{
	int i, k;

	for(i=0; i<V2S_NUM_ROWS; i++) {
		for(k=0; k<V2S_NUM_COLS; k++) {
			v2s[ch][i][k] = convolve_v2s_new_array(channel_list, num_channels, i, k);
			//printf("v2s[%d][%d]: %0.2f\n", i, k, v2s[i][k]);
		}
	}

	return 0;
}

static int init_v4s_filter_random(void)
{
	int i,j,k;
	float temp = 0.0;

	srandom(time(NULL));

	for(i=0; i<V4S_FILTER_NUM_CHANNELS; i++) {
		for(j=0; j<V4S_FILTER_ROWS; j++) {
			for(k=0; k<V4S_FILTER_COLS; k++) {
				temp =  (float) (random() % 50);
				v4s_filters[i][j][k] = temp / 50.0;
			}
		}
	}
	return 0;
}

static void get_feature_combinations(int ch, int *feature1, int *feature2)
{
	switch (ch)
	{
	case 0:
		*feature1 = 0;
		*feature2 = 1;
		break;
	case 1:
		*feature1 = 0;
		*feature2 = 2;
		break;
	case 2:
		*feature1 = 0;
		*feature2 = 3;
		break;
	case 3:
		*feature1 = 0;
		*feature2 = 4;
		break;
	case 4:
		*feature1 = 1;
		*feature2 = 2;
		break;
	case 5:
		*feature1 = 1;
		*feature2 = 3;
		break;
	case 6:
		*feature1 = 1;
		*feature2 = 4;
		break;
	case 7:
		*feature1 = 2;
		*feature2 = 3;
		break;
	case 8:
		*feature1 = 2;
		*feature2 = 4;
		break;
	case 9:
		*feature1 = 3;
		*feature2 = 4;
		break;
	default:
		//valiud values are only from 0 to 9.
		printf("Error: get_feature_combinations: ch value %d is invalid\n", ch);
	}
}

static int adjust_weights(int ch, int i, int j)
{
	int feature1, feature2;
	float inc_value = 0.2;
	float dec_value = 0.1;

	/*
	 * The feature combinations are static.
	 * They mention for each channel which 2 features are mated
	 */
	        
	get_feature_combinations(ch, &feature1, &feature2);

	/*
	 * If the two features occurred together then inrease their weight by a little bit
	 * If the two features didn;t occur together then decrease their weight by a little bit
	 * You can have a threshold. If both the features are above a certain value then you increase
	 */

	/*
	 * Where do we check if the two features occurred together?
	 * In the input image and/or the V2C features.
	 */

	/*
	 * So we will cehck if feature1 and feature2 occur together in the V2C and/or input image
	 * and if so we will increase the V4S[ch][i][j]
	 * If not we will decrease a little bit value from V4S[ch][i][j]
	 * The decrement will be a little less than the increment
	 * And when we do increment we will have to decrement others (maybe nearby ones)
	 * So that the weights don't saturate
	 */
	
	if(is_v2c_spatial_correlation_present(feature1, feature2, i, j))
	{
		v4s[ch][i][j] = v4s[ch][i][j] + inc_value;
		/* 
		 * ToDo: Decrement the value of other filters
		 * Find out which one can be decremented and do it
		 * So that weights don't all get high values eventually.
		 */
	}
	else
	{
		v4s[ch][i][j] = v4s[ch][i][j] - dec_value;
	}

	return 0;
}

/*
 * This is a new method of convolution we are using.
 * Here we have 10 filters for V4S. These 10 filters 
 * will tell us how to mate the 10 feature combinations that we have decided.
 * The filter tells us how the 2 features correlate.
 * For example if the filter is 
 *        0.1 0.0 0.1 0.0 0.0
 *        0.9 0.9 0.8 0.9 0.9
 *        0.0 0.1 0.2 0.0 0.1
 *        0.0 0.0 0.0 0.1 0.1
 *        0.0 0.1 0.1 0.1 0.0
 * It means that the correlation between the 2 features is the middle line
 * That we are interested in a new form where the we join the middle of two features together
 * This is just one simple example. The filter shape itself can be complex instead of linear.
 * And then the resulting new shape will be different
 * So we do the multiplication like this:
 * For each element of feature1 F1 and feature2 F2 and filtervalue FL
 * output += F1 * F2 * FL
 */

static float convolve_v4s(int feature1, int feature2, int c1, int c2, int filter_channel)
{
	float output = 0.0, value1 = 0.0, value2 = 0.0, filter;
	int row_start, col_start, row_end, col_end;
	int i, k;

	row_start = (c1*V4S_STRIDE) - V4S_PADDING;
	col_start = (c2*V4S_STRIDE) - V4S_PADDING;

	row_end = row_start + V4S_FILTER_ROWS;
	col_end = col_start + V4S_FILTER_COLS;

	for(i=row_start; i<row_end; i++) {
		for(k=col_start; k<col_end; k++) {
			/* this boundary check is helpful when we have padding */
			if(i < 0 || i >= V4S_NUM_ROWS || k < 0 || k >= V4S_NUM_COLS) {
				//printf("%s: i or k value invalid: i:%d k:%d\n", __FUNCTION__, i, k);
				value1 = 0.0;
				value2 = 0.0;
			}
			else {
				value1 = v2c[feature1][i][k];
				value2 = v2c[feature2][i][k];

				filter = v4s_filters[filter_channel][i][k];

				output += value1 * value2 * filter;
			}
		}
	}
	return output;
}

/* 
 * This function does the sliding and computes the convolution each time
 * it slides over the input
 */

/*
 * ch represents two things
 * The output v4s channel and 
 * The filter channel
 * They are both containing same values
 */

static int compute_v4s(int f1, int f2, int ch)
{
	int i, k;

        for(i=0; i<V4S_NUM_ROWS; i++) {
                for(k=0; k<V4S_NUM_COLS; k++) {
                        v4s[ch][i][k] = convolve_v4s(f1,f2, i, k, ch);
                        //printf("v4s[%d][%d][%d]: %0.2f\n", ch, i, k, v4s[ch][i][k]);
                }
        }

        return 0;
}

static void print_v4s_filter(void)
{
        int j,k,ch;

        for(ch=0; ch<V4S_FILTER_NUM_CHANNELS; ch++) {
                for(j=0; j<V4S_FILTER_ROWS; j++) {
                        for(k=0; k<V4S_FILTER_COLS; k++) {
                                printf("%f ", v4s_filters[ch][j][k]);
                        }
                        printf("\n");
                }
                printf("\n");
        }
}

/* This function is unused now */

static void construct_filter(float filter[][CS_NUM_COLS])
{
	int i, k;

	for(i=0; i<CS_NUM_ROWS; i++) {
		for(k=0; k<CS_NUM_COLS; k++) {
			filter[i][k] = -0.30; //this value s for surround, experiment with it
		}
	}

	//now initialize center with 1.25, it gives us a center-surrond filter
	//we assume filter size is 4 x 4, change below if filter size changes
	filter[1][1] = filter[1][2] = filter[2][1] = filter[2][2] = 1.25; //this value is for center, experiment with these values
}
