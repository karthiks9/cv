#!/usr/bin/python3

import zipfile
import struct
import os

# --- Configuration ---
# NOTE: Update these filenames if yours are different!
IMAGE_ZIP_FILE = 't10k-images.zip' 
LABEL_FILE = 't10k-labels-idx1-ubyte' # Standalone binary file

NUM_ITEMS = 10000
ROWS = 28
COLS = 28
PIXELS_PER_IMAGE = ROWS * COLS
MNIST_IMAGE_MAGIC = 2051 # 0x00000803
MNIST_LABEL_MAGIC = 2049 # 0x00000801

def parse_mnist_data():
    """
    Reads MNIST labels from a standalone binary file and images from a ZIP file, 
    verifies magic numbers, and outputs 10,000 text files named with their label.
    """
    
    # 1. Check for file existence using 'if'
    if not os.path.exists(IMAGE_ZIP_FILE):
        print(f"Error: Image ZIP file '{IMAGE_ZIP_FILE}' not found.")
        return
    if not os.path.exists(LABEL_FILE):
        print(f"Error: Label binary file '{LABEL_FILE}' not found.")
        return

    print("Starting to parse data from separate ZIP and binary files...")
    labels = []

    # --- PHASE 1: READ LABELS from binary file ---
    print(f"\nPhase 1: Reading labels from binary file '{LABEL_FILE}'...")
    
    # Open the standalone binary file directly
    with open(LABEL_FILE, 'rb') as f_labels:
        # Read the 8-byte header: (>II format: big-endian, 2 unsigned integers)
        header = f_labels.read(8)
        
        if len(header) < 8:
            print("Error: Could not read 8-byte header from label file.")
            return

        magic, count = struct.unpack('>II', header)

        # Check Magic Number for Labels
        if magic != MNIST_LABEL_MAGIC:
            print(f"Error: Invalid label magic number. Expected {MNIST_LABEL_MAGIC}, got {magic}.")
            return

        if count != NUM_ITEMS:
            print(f"Warning: Label file expected {NUM_ITEMS} items, found {count}.")

        # Read all remaining label bytes (1 byte per label)
        label_data = f_labels.read(count)
        
        if len(label_data) != count:
            print("Error: Could not read all label bytes.")
            return

        # Unpack the bytes into a list of integers (labels 0-9)
        labels = list(struct.unpack(f'{count}B', label_data))
        print(f"Successfully read {len(labels)} labels.")


    # --- PHASE 2: READ IMAGES from ZIP file and WRITE FILES ---
    print(f"\nPhase 2: Reading images from '{IMAGE_ZIP_FILE}' and writing output files...")
    
    with zipfile.ZipFile(IMAGE_ZIP_FILE, 'r') as zip_f_images:

        # Dynamically determine the internal image file name
        internal_files = zip_f_images.namelist()
        
        if not internal_files:
            print(f"Error: Image ZIP file '{IMAGE_ZIP_FILE}' is empty.")
            return

        # Use the name of the first file found inside the ZIP
        IMAGE_INTERNAL_NAME_ACTUAL = internal_files[0]
        print(f"Internal image file found: '{IMAGE_INTERNAL_NAME_ACTUAL}'")


        with zip_f_images.open(IMAGE_INTERNAL_NAME_ACTUAL, 'r') as f_images:
            
            # Read the 16-byte header: (>IIII format: big-endian, 4 unsigned integers)
            header = f_images.read(16)
            
            if len(header) < 16:
                print("Error: Could not read 16-byte header from image file.")
                return

            # Unpack the header (magic, count, rows, cols)
            magic, count, rows, cols = struct.unpack('>IIII', header)

            # Check Magic Number for Images
            if magic != MNIST_IMAGE_MAGIC:
                print(f"Error: Invalid image magic number. Expected {MNIST_IMAGE_MAGIC}, got {magic}.")
                return

            if count != NUM_ITEMS or rows != ROWS or cols != COLS:
                print("Warning: Image file header suggests non-standard dimensions or count.")
                
            # Iterate through all images
            for i in range(NUM_ITEMS):
                # Read all 784 bytes (pixels) for the current image
                image_data = f_images.read(PIXELS_PER_IMAGE)

                # Check for unexpected end of file
                if len(image_data) < PIXELS_PER_IMAGE:
                    print(f"Error: Reached end of image file unexpectedly at index {i}. Stopping.")
                    break

                # Unpack pixel values
                pixel_values = struct.unpack(f'{PIXELS_PER_IMAGE}B', image_data)

                # Get the corresponding label using the index 'i'
                current_label = labels[i]

                # Create the filename including the index and the label
                output_filename = f'image_{i:04d}_label_{current_label}.txt'

                # Build the 28x28 grid string for the output file
                output_content = []
                for r in range(ROWS):
                    row_pixels = pixel_values[r * COLS : (r + 1) * COLS]
                    # Convert numbers to strings and join with spaces
                    output_content.append(' '.join(map(str, row_pixels)))

                # Write the content to the new .txt file
                with open(output_filename, 'w') as out_f:
                    out_f.write('\n'.join(output_content))

                # Progress update
                if (i + 1) % 1000 == 0:
                    print(f"Processed and wrote image {i+1}/{NUM_ITEMS} (Index {i}, Label {current_label})")

    print("\nParsing complete. 10,000 labeled text files created in the script's directory.")

# Execute the function when the script runs
if __name__ == '__main__':
    parse_mnist_data()
