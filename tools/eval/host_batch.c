/*
 * Host batch runner for the generated model (B0 or B1 kernels, plain C on x86).
 *
 *   host_batch <out_tensor_idx> <list.txt> <out.bin>
 *
 * list.txt has one input file per line (raw int8, the model's input tensor). For each input the
 * model is run and the output tensor's bytes are appended to out.bin.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <out_tensor_idx> <list.txt> <out.bin>\n", argv[0]);
        return 2;
    }
    int out_idx = atoi(argv[1]);
    const model_tensor_t *in = NULL, *out = NULL;
    for (int i = 0; i < model_tensor_count; i++) {
        if (model_tensors[i].is_input) in = &model_tensors[i];
        if (model_tensors[i].idx == out_idx) out = &model_tensors[i];
    }
    if (!in || !out) {
        fprintf(stderr, "input/output tensor not found\n");
        return 2;
    }
    FILE *list = fopen(argv[2], "r");
    FILE *res = fopen(argv[3], "wb");
    if (!list || !res) return 2;

    int8_t *arena = model_arena();
    char path[1024];
    int n = 0;
    while (fgets(path, sizeof path, list)) {
        path[strcspn(path, "\r\n")] = 0;
        if (!path[0]) continue;
        FILE *f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "missing %s\n", path); return 2; }
        size_t got = fread(arena + in->offset, 1, (size_t)in->nbytes, f);
        fclose(f);
        if ((int)got != in->nbytes) { fprintf(stderr, "short read %s\n", path); return 2; }
        model_run();
        fwrite(arena + out->offset, 1, (size_t)out->nbytes, res);
        n++;
    }
    fclose(res);
    fclose(list);
    fprintf(stderr, "ran %d images\n", n);
    return 0;
}
