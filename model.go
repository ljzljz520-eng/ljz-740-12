package stablediffusion

import (
	"errors"
	"fmt"

	"github.com/example/stablediffusion/bindings"
)

// ModelOptions 描述加载模型上下文所需的最常用选项。
//
// 调用方只需关心四件事：模型路径、线程数、量化选项、是否使用 VAE。
// 更高级的参数（多模型路径、CPU offload、flash attention 等）
// 请使用 NewContext + ContextOptions。
type ModelOptions struct {
	// ModelPath 模型文件路径（必填），支持 .safetensors / .gguf
	ModelPath string

	// NThreads 推理线程数；<= 0 表示自动（按物理核心数）
	NThreads int

	// Wtype 权重量化类型，例如：
	//   bindings.SD_TYPE_F32  —— 不量化（零值）
	//   bindings.SD_TYPE_F16  —— 半精度（推荐默认）
	//   bindings.SD_TYPE_Q8_0 / SD_TYPE_Q4_K ... —— 量化加载，省显存/内存
	Wtype bindings.SdType

	// UseVAE 是否启用 VAE（把 latent 解码成图像所必需）。
	// 为 true 且 VaePath 非空时加载外部 VAE 文件；
	// 为 true 且 VaePath 为空时使用主模型内嵌的 VAE（若有）。
	UseVAE bool

	// VaePath 可选：外部 VAE 模型路径，仅在 UseVAE 为 true 时生效
	VaePath string
}

// DefaultModelOptions 返回一组合理默认选项：
// 自动线程数、F16 权重、启用 VAE（使用模型内嵌 VAE）。
func DefaultModelOptions(modelPath string) ModelOptions {
	return ModelOptions{
		ModelPath: modelPath,
		NThreads:  -1, // 自动
		Wtype:     bindings.SD_TYPE_F16,
		UseVAE:    true,
	}
}

// OpenModel 按给定选项加载模型，返回一个可关闭的模型上下文。
//
// 返回的 *Context 实现 io.Closer，使用完毕后必须调用 Close 释放资源，
// 推荐 defer：
//
//	model, err := stablediffusion.OpenModel(opts)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer model.Close() // 幂等，重复调用安全
func OpenModel(opts ModelOptions) (*Context, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("stablediffusion: ModelPath is required")
	}

	nThreads := opts.NThreads
	if nThreads == 0 {
		nThreads = -1 // 与 sd.cpp 约定一致：-1 表示自动
	}

	ctxOpts := ContextOptions{
		ModelPath:  opts.ModelPath,
		NThreads:   nThreads,
		Wtype:      opts.Wtype,
		EnableMmap: true,
	}
	if opts.UseVAE && opts.VaePath != "" {
		ctxOpts.VaePath = opts.VaePath
	}

	ctx, err := NewContext(ctxOpts)
	if err != nil {
		return nil, fmt.Errorf("stablediffusion: open model %q: %w", opts.ModelPath, err)
	}
	return ctx, nil
}
