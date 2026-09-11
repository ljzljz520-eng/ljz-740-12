package stablediffusion

import (
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/example/stablediffusion/bindings"
)

// newFakeContext 构造一个带假底层指针的 Context。
// 测试环境没有真实动态库，bindings 走 mock 实现，freeSdCtx 是 no-op，
// 因此对假指针调用 Close 是安全的。
func newFakeContext() *Context {
	return &Context{ctx: &bindings.SdCtx{}}
}

// 重复 Close 不应崩溃，且每次都返回 nil
func TestCloseIdempotent(t *testing.T) {
	c := newFakeContext()

	for i := 0; i < 3; i++ {
		if err := c.Close(); err != nil {
			t.Fatalf("第 %d 次 Close 返回错误: %v", i+1, err)
		}
	}

	if !c.closed {
		t.Fatal("Close 后 closed 标志应为 true")
	}
	if c.ctx != nil {
		t.Fatal("Close 后底层 ctx 应置为 nil")
	}
}

// 并发 Close 不应崩溃、不应 data race（配合 go test -race 验证）
func TestCloseConcurrent(t *testing.T) {
	c := newFakeContext()

	const goroutines = 64
	var wg sync.WaitGroup
	errs := make(chan error, goroutines)

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- c.Close()
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			t.Fatalf("并发 Close 返回错误: %v", err)
		}
	}
	if !c.closed {
		t.Fatal("并发 Close 后 closed 标志应为 true")
	}
}

// 已关闭的 Context 上执行操作应返回 ErrContextClosed，而不是崩溃
func TestUseAfterClose(t *testing.T) {
	c := newFakeContext()
	if err := c.Close(); err != nil {
		t.Fatalf("Close 失败: %v", err)
	}

	if _, err := c.GenerateImage(GenerationConfig{Width: 512, Height: 512, BatchCount: 1}); !errors.Is(err, ErrContextClosed) {
		t.Fatalf("GenerateImage 应返回 ErrContextClosed，实际: %v", err)
	}

	if _, err := c.GenerateVideo(VideoGenerationConfig{Width: 512, Height: 512}); !errors.Is(err, ErrContextClosed) {
		t.Fatalf("GenerateVideo 应返回 ErrContextClosed，实际: %v", err)
	}

	// 不返回 error 的取值方法应返回零值而不是把 nil 传给 C 层
	if got := c.GetDefaultSampleMethod(); got != 0 {
		t.Fatalf("关闭后 GetDefaultSampleMethod 应返回零值，实际: %v", got)
	}
	if got := c.GetDefaultScheduler(bindings.EULER_SAMPLE_METHOD); got != 0 {
		t.Fatalf("关闭后 GetDefaultScheduler 应返回零值，实际: %v", got)
	}
}

// 旧接口 Free 委托给 Close，同样幂等
func TestFreeAliasIdempotent(t *testing.T) {
	c := newFakeContext()
	c.Free()
	c.Free() // 第二次不应崩溃

	if _, err := c.GenerateImage(GenerationConfig{}); !errors.Is(err, ErrContextClosed) {
		t.Fatalf("Free 后 GenerateImage 应返回 ErrContextClosed，实际: %v", err)
	}
}

// OpenModel 校验必填参数
func TestOpenModelRequiresModelPath(t *testing.T) {
	if _, err := OpenModel(ModelOptions{}); err == nil {
		t.Fatal("空 ModelPath 应返回错误")
	}
}

// OpenModel 在模型加载失败时返回带路径信息的错误（mock 模式下 newSdCtx 返回 nil）
func TestOpenModelLoadFailure(t *testing.T) {
	_, err := OpenModel(ModelOptions{
		ModelPath: "not-exist-model.gguf",
		NThreads:  4,
		Wtype:     bindings.SD_TYPE_Q8_0,
		UseVAE:    true,
	})
	if err == nil {
		t.Fatal("mock 模式下 OpenModel 应返回错误")
	}
	if !strings.Contains(err.Error(), "not-exist-model.gguf") {
		t.Fatalf("错误信息应包含模型路径，实际: %v", err)
	}
}

// DefaultModelOptions 应给出合理默认值
func TestDefaultModelOptions(t *testing.T) {
	opts := DefaultModelOptions("models/m.gguf")
	if opts.ModelPath != "models/m.gguf" {
		t.Fatalf("ModelPath 不匹配: %q", opts.ModelPath)
	}
	if opts.NThreads != -1 {
		t.Fatalf("NThreads 默认应为自动(-1)，实际: %d", opts.NThreads)
	}
	if opts.Wtype != bindings.SD_TYPE_F16 {
		t.Fatalf("Wtype 默认应为 SD_TYPE_F16，实际: %v", opts.Wtype)
	}
	if !opts.UseVAE {
		t.Fatal("UseVAE 默认应为 true")
	}
}
