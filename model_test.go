package stablediffusion

import (
	"errors"
	"sync"
	"testing"

	"github.com/example/stablediffusion/bindings"
)

func testOpts() ModelOptions {
	return ModelOptions{
		ModelPath:    "models/test-model.gguf",
		Threads:      4,
		Quantization: bindings.SD_TYPE_Q8_0,
		UseVAE:       true,
	}
}

// openOrSkip 在 mock 环境下正常创建；若环境加载了真实动态库但没有模型文件，则跳过。
func openOrSkip(t *testing.T, opts ModelOptions) *Model {
	t.Helper()
	m, err := OpenModel(opts)
	if err != nil {
		t.Skipf("native library/model unavailable: %v", err)
	}
	return m
}

func TestOpenModelValidation(t *testing.T) {
	if _, err := OpenModel(ModelOptions{}); !errors.Is(err, ErrEmptyModelPath) {
		t.Fatalf("expected ErrEmptyModelPath, got %v", err)
	}

	bad := testOpts()
	bad.Quantization = bindings.SD_TYPE_COUNT
	if _, err := OpenModel(bad); !errors.Is(err, ErrInvalidQuant) {
		t.Fatalf("expected ErrInvalidQuant, got %v", err)
	}
}

func TestOpenModelFields(t *testing.T) {
	m := openOrSkip(t, testOpts())
	defer m.Close()

	if m.ModelPath() != "models/test-model.gguf" {
		t.Fatalf("unexpected model path: %s", m.ModelPath())
	}
	if m.Threads() != 4 {
		t.Fatalf("unexpected threads: %d", m.Threads())
	}
	if m.Quantization() != bindings.SD_TYPE_Q8_0 {
		t.Fatalf("unexpected quant: %v", m.Quantization())
	}
	if !m.UseVAE() {
		t.Fatal("expected UseVAE() == true")
	}
}

func TestModelCloseIdempotent(t *testing.T) {
	m := openOrSkip(t, testOpts())

	// 重复 Close 不应崩溃也不应报错
	for i := 0; i < 3; i++ {
		if err := m.Close(); err != nil {
			t.Fatalf("Close #%d returned error: %v", i, err)
		}
	}
	if !m.Closed() {
		t.Fatal("expected Closed() == true after Close")
	}
}

func TestModelCloseConcurrent(t *testing.T) {
	m := openOrSkip(t, testOpts())

	// 并发 Close：配合 -race 检查数据竞争；native 释放只能发生一次。
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.Close(); err != nil {
				t.Errorf("concurrent Close: %v", err)
			}
		}()
	}
	wg.Wait()
}

func TestModelUseAfterClose(t *testing.T) {
	m := openOrSkip(t, testOpts())
	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := m.GenerateImage(GenerationConfig{Prompt: "a cat"}); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed after close, got %v", err)
	}
}

func TestModelUseVAEFlag(t *testing.T) {
	opts := testOpts()
	opts.UseVAE = false
	m := openOrSkip(t, opts)
	defer m.Close()

	if m.UseVAE() {
		t.Fatal("expected UseVAE() == false")
	}
	if _, err := m.GenerateImage(GenerationConfig{Prompt: "a cat"}); !errors.Is(err, ErrVAEDisabled) {
		t.Fatalf("expected ErrVAEDisabled, got %v", err)
	}
}

func TestContextFreeAndCloseIdempotent(t *testing.T) {
	ctx, err := NewContext(DefaultContextOptions("models/test-model.gguf"))
	if err != nil {
		t.Skipf("native library/model unavailable: %v", err)
	}

	ctx.Free()                          // 旧接口先释放
	if err := ctx.Close(); err != nil { // 再调 Close：必须安全
		t.Fatalf("Close after Free returned error: %v", err)
	}
	ctx.Free() // 再次旧接口：仍然安全
	if !ctx.Closed() {
		t.Fatal("expected Closed() == true")
	}
	if _, err := ctx.GenerateImage(GenerationConfig{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
}
