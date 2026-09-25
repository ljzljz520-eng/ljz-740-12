package stablediffusion

import (
	"errors"
	"fmt"

	"github.com/example/stablediffusion/bindings"
)

// 模型生命周期相关的错误，可用 errors.Is 判断。
var (
	// ErrEmptyModelPath 模型路径为空。
	ErrEmptyModelPath = errors.New("stablediffusion: model path is empty")
	// ErrInvalidQuant 量化类型非法。
	ErrInvalidQuant = errors.New("stablediffusion: invalid quantization type")
	// ErrClosed 上下文已关闭，禁止继续使用。
	ErrClosed = errors.New("stablediffusion: context is closed")
	// ErrVAEDisabled 创建时指定 UseVAE=false，禁止需要 VAE 解码的操作。
	ErrVAEDisabled = errors.New("stablediffusion: VAE decode disabled (UseVAE=false)")
)

// ModelOptions 是加载一个模型所需的最小配置。
type ModelOptions struct {
	// ModelPath 模型文件路径（必填），如 models/sd-v1.5.gguf。
	ModelPath string
	// Threads 推理线程数；<=0 表示自动（由 native 层按物理核心数决定）。
	Threads int
	// Quantization 权重量化类型，见 bindings.SdType（如 SD_TYPE_Q8_0、SD_TYPE_F16）。
	Quantization bindings.SdType
	// UseVAE 是否允许使用 VAE 解码出像素图。
	// 注意：底层 stable-diffusion.cpp 没有“禁用 VAE”的开关（模型内置 VAE 总会加载），
	// 该开关由本封装在 Go 层强制执行：为 false 时 GenerateImage 返回 ErrVAEDisabled。
	UseVAE bool
}

// Model 是一个已加载的模型上下文，实现了 io.Closer。
//
// 使用完毕后必须调用 Close 释放 native 资源；Close 是幂等且并发安全的，
// 重复调用不会重复释放、也不会 panic，可放心配合 defer 使用：
//
//	m, err := stablediffusion.OpenModel(opts)
//	if err != nil { ... }
//	defer m.Close()
type Model struct {
	ctx  *Context
	opts ModelOptions
}

// OpenModel 按给定选项创建模型上下文。
//
// 返回的 *Model 持有 native 资源，调用方负责 Close（建议 defer）。
// 即使忘记 Close，GC 兜底 finalizer 也会释放 native 资源，但释放时机不确定，
// 不应依赖。
func OpenModel(opts ModelOptions) (*Model, error) {
	if opts.ModelPath == "" {
		return nil, ErrEmptyModelPath
	}
	if opts.Quantization < bindings.SD_TYPE_F32 || opts.Quantization >= bindings.SD_TYPE_COUNT {
		return nil, fmt.Errorf("%w: %d", ErrInvalidQuant, int(opts.Quantization))
	}

	threads := opts.Threads
	if threads <= 0 {
		threads = -1 // native 层约定：-1 表示自动选择线程数
	}

	ctxOpts := DefaultContextOptions(opts.ModelPath)
	ctxOpts.NThreads = threads
	ctxOpts.Wtype = opts.Quantization

	ctx, err := NewContext(ctxOpts)
	if err != nil {
		return nil, fmt.Errorf("stablediffusion: open model %q: %w", opts.ModelPath, err)
	}

	return &Model{ctx: ctx, opts: opts}, nil
}

// Close 释放模型占用的 native 资源。
//
// 重复调用是安全的（幂等）：第一次调用真正释放，后续调用直接返回 nil。
// 对 nil *Model 调用 Close 同样安全。
func (m *Model) Close() error {
	if m == nil || m.ctx == nil {
		return nil
	}
	return m.ctx.Close()
}

// Closed 报告模型是否已关闭（资源已释放）。
func (m *Model) Closed() bool {
	if m == nil || m.ctx == nil {
		return true
	}
	return m.ctx.Closed()
}

// ModelPath 返回创建时使用的模型路径。
func (m *Model) ModelPath() string { return m.opts.ModelPath }

// Threads 返回创建时指定的线程数（<=0 表示自动）。
func (m *Model) Threads() int { return m.opts.Threads }

// Quantization 返回创建时指定的权重量化类型。
func (m *Model) Quantization() bindings.SdType { return m.opts.Quantization }

// UseVAE 返回创建时指定的是否允许 VAE 解码。
func (m *Model) UseVAE() bool { return m.opts.UseVAE }

// Context 返回底层上下文，用于采样器选择等高级操作。
// 不要直接 Close 返回的上下文，统一通过 Model.Close 释放。
func (m *Model) Context() *Context { return m.ctx }

// GenerateImage 生成图像。
//
// 若创建时 UseVAE=false，返回 ErrVAEDisabled；模型已关闭时返回 ErrClosed。
func (m *Model) GenerateImage(cfg GenerationConfig) ([]*Image, error) {
	if m == nil || m.ctx == nil {
		return nil, ErrClosed
	}
	if !m.opts.UseVAE {
		return nil, ErrVAEDisabled
	}
	return m.ctx.GenerateImage(cfg)
}
