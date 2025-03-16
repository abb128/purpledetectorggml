all: inference inference-vk quantize classic

run: inference
	LD_LIBRARY_PATH=. ./inference

inference: 6-ggml-inference.c ggml/build/src/libggml.a
	g++ 6-ggml-inference.c -o inference -Iggml/include -Lggml/build/src ggml/build/src/libggml.a ggml/build/src/libggml-base.a ggml/build/src/libggml-cpu.a -Os

inference-vk: 8-ggml-inference-vk.c ggml/build/src/libggml.a
	g++ 8-ggml-inference-vk.c -o inference-vk -Iggml/include -Lggml/build/src ggml/build/src/libggml.a ggml/build/src/libggml-base.a ggml/build/src/libggml-cpu.a ggml/build/src/ggml-vulkan/libggml-vulkan.a -lvulkan -Os

quantize: 7-ggml-quantization.c ggml/build/src/libggml.a
	g++ 7-ggml-quantization.c -o quantize -Iggml/include -Lggml/build/src ggml/build/src/libggml.a ggml/build/src/libggml-base.a ggml/build/src/libggml-cpu.a

classic: 0-basic-algorithm.c
	g++ 0-basic-algorithm.c -o classic -Os

ggml/build/src/libggml.a:
	cmake -S ggml -B ggml/build -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
	cmake --build ggml/build -- -j8

clean:
	rm inference quantize classic ||:
	rm -rf ggml/build ||: