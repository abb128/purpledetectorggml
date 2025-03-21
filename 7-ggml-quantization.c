#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

bool file_exists(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

bool ends_with(const std::string &a, const std::string &b) {
    if (b.length() > a.length()) {
        return false;
    }
    return std::equal(b.rbegin(), b.rend(), a.rbegin());
}

int quantize_model(
    const std::string &fname_inp,
    const std::string &fname_out,
    ggml_type ftype
) {
    size_t ctx_size = 512 * 1024 * 1024;

    struct ggml_init_params params {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct gguf_init_params ggufparams = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx,
    };

    struct gguf_context * gctx = gguf_init_from_file(fname_inp.c_str(), ggufparams);

    struct gguf_context * goutctx = gguf_init_empty();

    std::vector<std::string> u32_names = {
        "input_width",
        "layer_count"
    };

    for(std::string name : u32_names)
        gguf_set_val_u32(goutctx, name.c_str(), gguf_get_val_u32(gctx, gguf_find_key(gctx, name.c_str())));

    gguf_set_val_str(goutctx, "general.architecture", gguf_get_val_str(gctx, gguf_find_key(gctx, "general.architecture")));
    gguf_set_val_str(goutctx, "general.name",         gguf_get_val_str(gctx, gguf_find_key(gctx, "general.name")));
    gguf_set_val_str(goutctx, "general.description",  gguf_get_val_str(gctx, gguf_find_key(gctx, "general.description")));

    std::vector<uint8_t> read_data;
    std::vector<uint8_t> work;

    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = std::string(cur->name);
        if(name.size() == 0) continue;
        if(name == "GGUF tensor data binary blob") continue;

        //bool quantize = ends_with(name, "weight") || ends_with(name, "bias");
        bool quantize = ends_with(name, "weight");

        if(ftype == GGML_TYPE_F32) quantize = false;
        
        enum ggml_type new_type;
        void *new_data;
        size_t new_size = 0;

        if(!quantize) {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        } else {
            const int64_t nelements = ggml_nelements(cur);

            float *f32_data;
            if(cur->type == GGML_TYPE_F32) {
                f32_data = (float *)cur->data;
            } else {
                printf("Tensor [%s] is non-float (%s)! Requantizing a file is not supported.\n", name.c_str(), ggml_type_name(cur->type));
                exit(1);
            }

            new_data = malloc(nelements * 4);

            const int64_t n_per_row = cur->ne[0];
            const int64_t nrows = cur->ne[1];

            printf("Quantizing tensor %s (%d, %d, %d, %d)\n", name.c_str(), cur->ne[3], cur->ne[2], cur->ne[1], cur->ne[0]);
            new_size += ggml_quantize_chunk(ftype, f32_data, new_data, 0, nrows, n_per_row, nullptr);
            new_type = ftype;
        }

        printf("[%c] Adding tensor %s (%8.3f MB)\n", quantize ? 'Q' : ' ', name.c_str(), new_size / 1024.0 / 1024.0);
        gguf_add_tensor(goutctx, cur);
        gguf_set_tensor_type(goutctx, name.c_str(), new_type);
        gguf_set_tensor_data(goutctx, name.c_str(), new_data);
    }

    gguf_write_to_file(goutctx, fname_out.c_str(), /*only_meta =*/ false);

    return 0;
}
    
static const std::map<std::string, enum ggml_type> TYPE_MAP = {
    {"q8_k",    GGML_TYPE_Q8_K   },
    {"q6_k",    GGML_TYPE_Q6_K   },
    {"q5_k",    GGML_TYPE_Q5_K   },
    {"q4_k",    GGML_TYPE_Q4_K   },
    {"q3_k",    GGML_TYPE_Q3_K   },
    {"q2_k",    GGML_TYPE_Q2_K   },

    {"f16",     GGML_TYPE_F16    },

    {"q4_0",    GGML_TYPE_Q4_0   },
    {"q4_1",    GGML_TYPE_Q4_1   },
    {"q5_0",    GGML_TYPE_Q5_0   },
    {"q5_1",    GGML_TYPE_Q5_1   },
    {"q8_0",    GGML_TYPE_Q8_0   },
    {"q8_1",    GGML_TYPE_Q8_1   },
    
    {"f32",     GGML_TYPE_F32    },
    {"iq2_xxs", GGML_TYPE_IQ2_XXS},
    {"iq2_xs",  GGML_TYPE_IQ2_XS },
    {"iq3_xxs", GGML_TYPE_IQ3_XXS},
    {"iq1_s",   GGML_TYPE_IQ1_S  },
    {"iq4_nl",  GGML_TYPE_IQ4_NL },
    {"iq3_s",   GGML_TYPE_IQ3_S  },
    {"iq2_s",   GGML_TYPE_IQ2_S  },
    {"iq4_xs",  GGML_TYPE_IQ4_XS },
    {"i8",      GGML_TYPE_I8     },
    {"i16",     GGML_TYPE_I16    },
    {"i32",     GGML_TYPE_I32    },
    {"i64",     GGML_TYPE_I64    },
    {"f64",     GGML_TYPE_F64    },
    {"iq1_m",   GGML_TYPE_IQ1_M  },
    {"bf16",    GGML_TYPE_BF16   },
    {"tq1_0",   GGML_TYPE_TQ1_0  },
    {"tq2_0",   GGML_TYPE_TQ2_0  }
};

int print_usage(void){
    printf("usage: ./quantize source.gguf target.gguf q4_k\n");

    printf("available types:\n");
    for(const auto &entry : TYPE_MAP) {
        printf(" * %s\n", entry.first.c_str());
    }

    exit(1);

    return 1;
}

int main(int argc, char ** argv) {
    if(argc != 4) return print_usage();

    std::string src_filename   = argv[1];
    std::string out_filename   = argv[2];
    std::string quant_typename = argv[3];

    if(!file_exists(src_filename.c_str())) {
        printf("file %s not found\n", src_filename.c_str());
        return 1;
    }

    auto val = TYPE_MAP.find(quant_typename);
    if(val == TYPE_MAP.end()) return print_usage();

    enum ggml_type t = val->second;
    printf("quantizing to type %s\n", ggml_type_name(t));

    ggml_time_init();
    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    return quantize_model(src_filename, out_filename, t);
}
