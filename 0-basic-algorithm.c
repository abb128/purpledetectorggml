#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

float *load_image(const char *path) {
    float *buffer = (float *)calloc(sizeof(float), 128 * 64 * 3);
    FILE *file = fopen(path, "rb");
    assert(fread(buffer, sizeof(float), 128 * 64 * 3, file) == 128*64*3);
    assert(fclose(file) == 0);
    return buffer;
}


float is_purple(const float *image, int columns, int rows) {
    float total_r = 0.0f;
    float total_g = 0.0f;
    float total_b = 0.0f;
    for(int y=0; y<columns; y++){
        for(int x=0; x<rows; x++) {
            total_r += image[0];
            total_g += image[1];
            total_b += image[2];

            image += 3;
        }
    }

    total_r /= columns * rows;
    total_g /= columns * rows;
    total_b /= columns * rows;

    float result = (total_r + total_b - total_g*2.0f) * 2.0f;
    if(result < 0.0f) result = 0.0f;
    if(result > 1.0f) result = 1.0f;

    return result;
}

int main() {
    const float *purple_img = load_image("test_dataset/purple/Dropped Image (6).bin");
    const float *nonpurple_img = load_image("test_dataset/nonpurple/Dropped Image (6).bin");

    printf("Purple image result: %.2f\n", is_purple(purple_img, 128, 64));
    printf("Nonpurple image result: %.2f\n", is_purple(nonpurple_img, 128, 64));
}