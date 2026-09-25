// 示例：模型上下文的创建与释放
//
// 演示如何用 OpenModel 传入「模型路径 / 线程数 / 量化选项 / 是否使用 VAE」
// 四个参数加载模型，并通过 defer model.Close() 保证底层资源被释放。
//
// 运行（需要真实模型文件和编译好的 libstable-diffusion）：
//
//	# 量化类型通过环境变量演示，默认 f16；可选 f32 / q8_0 / q4_k 等
//	SD_LIB_PATH=./stable-diffusion.cpp/build/bin/libstable-diffusion.so \
//	go run ./examples/model ./models/model.gguf
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	sd "github.com/example/stablediffusion"
	"github.com/example/stablediffusion/bindings"
)

func parseQuantization(name string) (bindings.SdType, error) {
	t := bindings.StrToSdType(strings.ToLower(name))
	if t == bindings.SD_TYPE_COUNT {
		return 0, fmt.Errorf("unsupported quantization type %q", name)
	}
	return t, nil
}

func main() {
	// 位置参数：模型路径；flag：线程数、量化类型、是否使用 VAE
	threads := flag.Int("threads", 0, "推理线程数，0 或负数表示自动选择（物理核心数）")
	quant := flag.String("quant", "f16", "权重量化/精度类型，例如 f32 / f16 / q8_0 / q4_k")
	noVAE := flag.Bool("no-vae", false, "不加载完整 VAE，仅输出 TAESD 快速预览")
	flag.Parse()

	modelPath := flag.Arg(0)
	if modelPath == "" {
		modelPath = os.Getenv("MODEL_PATH")
	}
	if modelPath == "" {
		log.Fatal("用法: model [flags] <模型路径>，或设置 MODEL_PATH 环境变量")
	}

	wtype, err := parseQuantization(*quant)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("System: %s\n", sd.GetSystemInfo())
	fmt.Printf("加载模型: path=%s threads=%d quant=%s useVAE=%t\n",
		modelPath, *threads, sd.GetTypeName(wtype), !*noVAE)

	// 1) 创建模型上下文：调用方只传模型路径、线程数、量化选项、是否使用 VAE
	model, err := sd.OpenModel(modelPath, *threads, wtype, !*noVAE)
	if err != nil {
		log.Fatalf("加载模型失败: %v", err)
	}

	// 2) defer 释放资源：函数退出时自动调用 Close
	//    Close 可重复调用，不会崩溃，因此即使下面再手动 Close 一次也是安全的
	defer func() {
		if err := model.Close(); err != nil {
			log.Printf("关闭模型时出错: %v", err)
		}
		log.Println("模型资源已通过 defer 释放")
	}()

	// 3) 通过内嵌的 Context 使用模型
	method := model.GetDefaultSampleMethod()
	scheduler := model.GetDefaultScheduler(method)
	fmt.Printf("默认采样器: method=%s scheduler=%s\n",
		sd.GetSampleMethodName(method), sd.GetSchedulerName(scheduler))

	// 后续可在同一个 model 上反复执行生成：
	//
	//   images, err := model.GenerateImage(sd.GenerationConfig{
	//       Prompt:     "a cute corgi, studio lighting",
	//       Width:      512,
	//       Height:     512,
	//       BatchCount: 1,
	//       Sampler: sd.SamplerConfig{
	//           Scheduler: scheduler,
	//           Method:    method,
	//           Steps:     20,
	//           TxtCfg:    7.5,
	//       },
	//   })
	//   ...
	_ = scheduler

	// 4) 手动提前释放也是安全的；defer 中的第二次 Close 不会崩溃（幂等）
	if err := model.Close(); err != nil {
		log.Printf("手动关闭模型时出错: %v", err)
	}
	fmt.Printf("Closed() = %t\n", model.Closed())

	// 关闭后再使用会拿到明确的错误，而不是段错误
	if _, err := model.GenerateImage(sd.GenerationConfig{}); err != nil {
		fmt.Printf("已关闭的模型拒绝生成: %v\n", err)
	}

	// 甚至在 nil 对象上调用 Close 也不会 panic
	var nilModel *sd.Model
	_ = nilModel.Close()
	fmt.Println("nil 模型 Close 安全")
}
