#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <vector>


#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include "gguf.h"

#define PRINT_SHAPE(t, text) printf(text " shape: %d %d %d %d\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3])

struct purple_detection_model {
    const char *name;
    const char *description;

    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    ggml_gallocr_t allocr;

    struct ggml_tensor *linear_0_weight;
    struct ggml_tensor *linear_0_bias;
    struct ggml_tensor *linear_2_weight;
    struct ggml_tensor *linear_2_bias;

    struct ggml_context *ctx;
};


// Utility to upload model weights to backend
// 1. Create tensor_uploader and a no_alloc ctx
// 2. Call mark_upload_tensor on all necessary tensors, it will replace them
// 3. Call ggml_backend_alloc_ctx_tensors and allocate the new tensors
// 4. Call upload_marked_tensors to upload the weights
struct tensor_uploader {
    struct ggml_tensor *from[256];
    struct ggml_tensor *to[256];
    size_t head;
};

void mark_upload_tensor(struct tensor_uploader *uploader, struct ggml_context *ctx, struct ggml_tensor **tensor) {
    struct ggml_tensor *orig = *tensor;

    *tensor = ggml_new_tensor_4d(ctx, orig->type, orig->ne[0], orig->ne[1], orig->ne[2], orig->ne[3]);

    uploader->from[uploader->head] = orig;
    uploader->to[uploader->head] = *tensor;

    uploader->head++;
}

void upload_marked_tensors(struct tensor_uploader *uploader){
    for(int i=0; i<uploader->head; i++){
        assert(ggml_nbytes(uploader->from[i]) == ggml_nbytes(uploader->to[i]));

        ggml_backend_tensor_set(
            /*    dst */ uploader->to[i],
            /*    src */ uploader->from[i]->data,
            /* offset */ 0,
            /*   size */ ggml_nbytes(uploader->from[i])
        );
    }
}

struct ggml_tensor *run_model(
    struct ggml_context *ctx,
    struct purple_detection_model *model,
    struct ggml_tensor *img
) {
    struct ggml_tensor *x = img;
    x = ggml_mul_mat(ctx, model->linear_0_weight, x);
    x = ggml_add(ctx, x, model->linear_0_bias);
    x = ggml_sigmoid(ctx, x);

    x = ggml_mul_mat(ctx, model->linear_2_weight, x);
    x = ggml_add(ctx, x, model->linear_2_bias);
    x = ggml_sigmoid(ctx, x);
    return x;
}


float *load_image(const char *path) {
    float *buffer = (float *)calloc(sizeof(float), 128 * 64 * 3);
    FILE *file = fopen(path, "rb");
    assert(fread(buffer, sizeof(float), 128 * 64 * 3, file) == 128*64*3);
    assert(fclose(file) == 0);
    return buffer;
}

struct ggml_cgraph *run_model_with_graph(
    int batch_size,
    struct purple_detection_model *model
);

struct purple_detection_model *load_model(const char *path){
    struct purple_detection_model *model = (struct purple_detection_model *)
            calloc(sizeof(struct purple_detection_model), 1);

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context *ctx = ggml_init(params);

    model->ctx = ctx;
    model->backend = ggml_backend_vk_init(0);
    model->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model->backend));

    struct ggml_init_params params2 = {
        /*.mem_size   =*/ 1,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context *pctx = ggml_init(params2);
    

    struct gguf_init_params ggufparams = {
        /*.no_alloc =*/  false,
        /*.ctx      =*/  &pctx,
    };
    struct gguf_context * gctx = gguf_init_from_file(path, ggufparams);

    const char *arch = gguf_get_val_str(gctx, gguf_find_key(gctx, "general.architecture"));
    assert(strcmp(arch, "purpledetector") == 0);

    assert(gguf_get_val_u32(gctx, gguf_find_key(gctx, "input_width")) == 64);
    assert(gguf_get_val_u32(gctx, gguf_find_key(gctx, "layer_count")) == 2);

    model->name = gguf_get_val_str(gctx, gguf_find_key(gctx, "general.name"));
    model->description = gguf_get_val_str(gctx, gguf_find_key(gctx, "general.description"));


    for (struct ggml_tensor * cur = ggml_get_first_tensor(pctx);
            cur != NULL;
            cur = ggml_get_next_tensor(pctx, cur)) {
        if(strcmp(cur->name, "0.weight") == 0) {
            model->linear_0_weight = cur;
        }else if(strcmp(cur->name, "0.bias") == 0) {
            model->linear_0_bias = cur;
        }else if(strcmp(cur->name, "2.weight") == 0) {
            model->linear_2_weight = cur;
        }else if(strcmp(cur->name, "2.bias") == 0) {
            model->linear_2_bias = cur;
        }
    }

    assert(model->linear_0_weight != NULL);
    assert(model->linear_0_bias != NULL);
    assert(model->linear_2_weight != NULL);
    assert(model->linear_2_bias != NULL);

    struct tensor_uploader uploader = { 0 };
    mark_upload_tensor(&uploader, ctx, &model->linear_0_weight);
    mark_upload_tensor(&uploader, ctx, &model->linear_0_bias);
    mark_upload_tensor(&uploader, ctx, &model->linear_2_weight);
    mark_upload_tensor(&uploader, ctx, &model->linear_2_bias);

    model->buffer = ggml_backend_alloc_ctx_tensors(model->ctx, model->backend);
    upload_marked_tensors(&uploader);

    struct ggml_cgraph *gf = run_model_with_graph(128, model);
    ggml_gallocr_reserve(model->allocr, gf);

    return model;
}

struct ggml_cgraph *run_model_with_graph(
    int batch_size,
    struct purple_detection_model *model
) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporary context to build the graph
    struct ggml_context *ctx0 = ggml_init(params0);

    struct ggml_cgraph  *gf = ggml_new_graph(ctx0);

    struct ggml_tensor *inputs = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 64*3, batch_size);
    ggml_set_input(inputs);
    ggml_set_name(inputs, "inputs");

    struct ggml_tensor *out = run_model(ctx0, model, inputs);

    out = ggml_sum(ctx0, out);
    ggml_set_output(out);
    ggml_set_name(out, "output");
    
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);

    return gf;
}

float is_purple(
    const float *image,
    struct purple_detection_model *model
) {
    struct ggml_cgraph *gf = run_model_with_graph(128, model);
    ggml_gallocr_alloc_graph(model->allocr, gf);
    
    struct ggml_tensor * graph_input = ggml_graph_get_tensor(gf, "inputs");
    struct ggml_tensor * graph_output = ggml_graph_get_tensor(gf, "output");
    ggml_backend_tensor_set(
            /*    dst */ graph_input,
            /*    src */ image,
            /* offset */ 0,
            /*   size */ sizeof(float)*128*64*3
    );

    ggml_backend_graph_compute(model->backend, gf);

    float outputs[1];
    ggml_backend_tensor_get(
        /*    src */ graph_output,
        /*   dest */ outputs,
        /* offset */ 0,
        /*   size */ sizeof(float)
    );
    return outputs[0] / 128.0f;
}

int main(int argc, char *argv[]) {
    const float *purple_img = load_image("test_dataset/purple/Dropped Image (6).bin");
    const float *nonpurple_img = load_image("test_dataset/nonpurple/Dropped Image (6).bin");

    const char *model_path;
    if(argc == 2) {
        model_path = argv[1];
    } else {
        model_path = "weights.gguf";
    }
    
    struct purple_detection_model *model = load_model(model_path);

    printf("Model Name: %s\n", model->name);
    printf("Model Description: %s\n", model->description);

    printf("Purple image result: %.2f\n", is_purple(purple_img, model));
    printf("Nonpurple image result: %.2f\n", is_purple(nonpurple_img, model));

    // Test if it runs out of memory
    for(int i=0; i<9999; i++) is_purple(nonpurple_img, model);
    printf("All OK\n");
}