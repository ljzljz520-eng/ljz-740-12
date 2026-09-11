// 示例：模型上下文的创建与释放
//
// 演示：
//  1. 调用方只传模型路径、线程数、量化选项、是否使用 VAE；
//  2. OpenModel 返回可关闭对象（io.Closer）；
//  3. 使用 defer 释放资源；
//  4. Close 幂等——重复调用不会崩溃。
//
// 运行：
//
//	go run ./examples/model_context -model models/model.gguf
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	stablediffusion "github.com/example/stablediffusion"
	"github.com/example/stablediffusion/bindings"
)

func main() {
	modelPath := flag.String("model", "models/model.gguf", "模型文件路径（.gguf / .safetensors）")
	vaePath := flag.String("vae", "", "可选：外部 VAE 模型路径")
	threads := flag.Int("threads", 4, "推理线程数，<=0 表示自动")
	quantize := flag.String("quantize", "f16", "权重量化类型：f32 / f16 / q8_0 / q4_k ...")
	flag.Parse()

	if _, err := os.Stat(*modelPath); err != nil {
		log.Fatalf("模型文件不可读: %v", err)
	}

	// 1) 组装选项：模型路径、线程数、量化选项、是否使用 VAE
	wtype := bindings.StrToSdType(*quantize) // 字符串 -> 量化枚举
	if wtype == bindings.SD_TYPE_COUNT {
		log.Fatalf("未知的量化类型: %q（可选 f32 / f16 / q8_0 / q4_k ...）", *quantize)
	}
	opts := stablediffusion.ModelOptions{
		ModelPath: *modelPath,
		NThreads:  *threads,
		Wtype:     wtype,
		UseVAE:    true,
		VaePath:   *vaePath,
	}

	// 2) 创建模型上下文，返回可关闭对象
	model, err := stablediffusion.OpenModel(opts)
	if err != nil {
		log.Fatalf("创建模型上下文失败: %v", err)
	}

	// 3) defer 释放资源：函数返回时自动调用 model.Close()
	defer func() {
		if err := model.Close(); err != nil {
			log.Printf("关闭模型上下文失败: %v", err)
		}
		fmt.Println("模型上下文已释放 (defer Close)")
	}()

	fmt.Printf("模型已加载: %s (线程数=%d, 量化=%s, VAE=%v)\n",
		*modelPath, *threads, *quantize, opts.UseVAE)

	// ... 在这里使用 model 生成图像，例如：
	// imgs, err := model.GenerateImage(stablediffusion.GenerationConfig{...})

	// 4) Close 幂等演示：即使这里显式关闭一次，
	//    上面 defer 的第二次 Close 也安全，不会重复释放或崩溃。
	if err := model.Close(); err != nil {
		log.Fatalf("关闭模型上下文失败: %v", err)
	}
	fmt.Println("已显式调用 Close，defer 中的重复 Close 依然安全")
}
