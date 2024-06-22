/* Header file for retina, v1 */

#define IMAGE_NUM_PIXELS 28 /* image size is square(symmetrical rows, columns) */
#define CONES_PER_PIXEL (1)
#define CONES_NUM_ROWS (IMAGE_NUM_PIXELS)
#define CONES_NUM_COLS (IMAGE_NUM_PIXELS * CONES_PER_PIXEL)

/* parameters for calculation */
#define CS_NUM_ROWS (1) /* center surround rows, usually 4 */
#define CS_NUM_COLS (1) /* center surround cols, usually 4 */
#define STRIDE (1)
#define CONES_PADDING (0)
#define RGC_NUM_ROWS (((CONES_NUM_ROWS + (2 * CONES_PADDING) - CS_NUM_ROWS) / STRIDE) + 1)
#define RGC_NUM_COLS (((CONES_NUM_COLS + (2 * CONES_PADDING) - CS_NUM_COLS) / STRIDE) + 1)

#define IMAGE_FILE_NAME "inputs/image.txt"

/* Definitions for V1 variables, structures, etc */

#define V1_NUM_FILTERS 9
#define V1_FILTER_ROWS 3
#define V1_FILTER_COLS 3
#define V1_STRIDE (1) //this used to be 1
#define V1_PADDING (0)

#define V1S_NUM_ROWS (((RGC_NUM_ROWS  + (2 * V1_PADDING) - V1_FILTER_ROWS) / V1_STRIDE) + 1)
#define V1S_NUM_COLS (((RGC_NUM_COLS  + (2 * V1_PADDING) - V1_FILTER_COLS) / V1_STRIDE) + 1)
#define V1S_NUM_CHANNELS V1_NUM_FILTERS

/* we scale the pooling area only column-wise for now, maybe later we will extend to row too */
#define V1C_POOL_ROWS (2)
#define V1C_POOL_COLS (2)

#define V1C_NUM_ROWS (V1S_NUM_ROWS/V1C_POOL_ROWS)
#define V1C_NUM_COLS (V1S_NUM_COLS/V1C_POOL_COLS)
#define V1C_NUM_CHANNELS V1S_NUM_CHANNELS

#define V2S_NUM_CHANNELS 5 //right curve, left curve, top curve, bottom curve, circle
#ifdef V2S_FILTERS_3
#define V2S_FILTER_ROWS 3
#define V2S_FILTER_COLS 3
#else
#define V2S_FILTER_ROWS 5
#define V2S_FILTER_COLS 5
#endif

#define V2S_FILTER_CHANNELS V1C_NUM_CHANNELS
#define V2S_STRIDE (1)
#define V2S_PADDING (0)

#define V2S_NUM_ROWS (((V1C_NUM_ROWS  + (2 * V2S_PADDING) - V2S_FILTER_ROWS) / V2S_STRIDE) + 1)
#define V2S_NUM_COLS (((V1C_NUM_COLS  + (2 * V2S_PADDING) - V2S_FILTER_COLS) / V2S_STRIDE) + 1)

#define V2C_POOL_ROWS (2)
#define V2C_POOL_COLS (2)

#define V2C_NUM_ROWS (V2S_NUM_ROWS/V2C_POOL_ROWS)
#define V2C_NUM_COLS (V2S_NUM_COLS/V2C_POOL_COLS)
#define V2C_NUM_CHANNELS V2S_NUM_CHANNELS

/* ========================================================================== */

/* V4 defines */

/* simple defines */

// the number of channels here (10) is chosen by us, not derived
#define V4S_NUM_CHANNELS 10 /* 4 + 3 + 2 + 1 : We mate each channel in V2C with each other */
#define V4S_FILTER_ROWS 5
#define V4S_FILTER_COLS 5

// we have 10 filters because the number of feature combinations we have decided is 10

#define V4S_FILTER_NUM_CHANNELS 10 /* we want 10 filters of size 5*5 */
#define V4S_STRIDE (1)
#define V4S_PADDING (1)

#define V4S_NUM_ROWS (((V2C_NUM_ROWS + (2 * V4S_PADDING) - V4S_FILTER_ROWS) / V4S_STRIDE) + 1)
#define V4S_NUM_COLS (((V2C_NUM_COLS + (2 * V4S_PADDING) - V4S_FILTER_COLS) / V4S_STRIDE) + 1)

