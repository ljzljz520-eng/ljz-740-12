// 演示模型上下文的创建与释放：
//   - 调用方只需要传 模型路径 / 线程数 / 量化选项 / 是否使用 VAE
//   - 返回可关闭的对象（实现 io.Closer）
//   - defer 释放资源；重复 Close 安全，不会崩溃
package main

import (
	"fmt"
	"log"

	sd "github.com/example/stablediffusion"
	"github.com/example/stablediffusion/bindings"
)

func main() {
	// 1. 传参创建模型上下文
	model, err := sd.OpenModel(sd.ModelOptions{
		ModelPath:    "models/sd-v1-5.gguf", // 模型路径
		Threads:      4,                     // 推理线程数（<=0 表示自动）
		Quantization: bindings.SD_TYPE_Q8_0, // 权重量化类型
		UseVAE:       true,                  // 是否使用 VAE 解码
	})
	if err != nil {
		log.Fatalf("open model failed: %v", err)
	}

	// 2. defer 释放 native 资源，函数返回时必定执行。
	//    Close 是幂等的，即使下面显式再调一次也不会重复释放。
	defer func() {
		if err := model.Close(); err != nil {
			log.Printf("close: %v", err)
		}
	}()

	fmt.Printf("model ready: path=%s threads=%d quant=%s vae=%v\n",
		model.ModelPath(), model.Threads(),
		sd.GetTypeName(bindings.SdType(model.Quantization())), model.UseVAE())

	// 3. 正常使用（无真实模型/动态库时生成会失败，仅演示调用方式）
	if _, err := model.GenerateImage(sd.GenerationConfig{
		Prompt: "a cute cat, high quality",
		Width:  512,
		Height: 512,
	}); err != nil {
		log.Printf("generate (expected to fail in mock env): %v", err)
	}

	// 4. 提前显式关闭；defer 里那次 Close 会变成空操作，不会 panic
	if err := model.Close(); err != nil {
		log.Printf("close: %v", err)
	}
	fmt.Println("first Close done")

	// 5. 重复 Close 不应崩溃
	if err := model.Close(); err != nil {
		log.Printf("close again: %v", err)
	}
	fmt.Println("second Close is a no-op, still alive")
}
