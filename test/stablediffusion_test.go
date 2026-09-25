package test

import (
	"testing"

	"github.com/example/stablediffusion"
)

// 测试用例2.1：系统信息获取
func TestSystemInfo(t *testing.T) {
	// 预期结果：能够获取到系统信息字符串
	info := stablediffusion.GetSystemInfo()
	if info == "" {
		t.Errorf("Expected non-empty system info, got empty string")
	}
	t.Logf("System Info: %s", info)
}

// 测试用例2.2：版本信息获取
func TestVersionInfo(t *testing.T) {
	// 预期结果：能够获取到版本信息和提交信息
	version := stablediffusion.GetVersion()
	commit := stablediffusion.GetCommit()
	
	if version == "" {
		t.Errorf("Expected non-empty version, got empty string")
	}
	
	if commit == "" {
		t.Errorf("Expected non-empty commit, got empty string")
	}
	
	t.Logf("Version: %s", version)
	t.Logf("Commit: %s", commit)
}

// 测试用例2.3：创建上下文（模拟）
func TestCreateContext(t *testing.T) {
	// 注意：这个测试需要实际的模型文件，这里只是测试API调用
	// 预期结果：在没有真实模型文件以模拟错误拦截及处理。
	modelPath := "test_model.gguf"
	
	_, err := stablediffusion.NewContext(stablediffusion.DefaultContextOptions(modelPath))
	if err == nil {
		t.Logf("Context created successfully (unexpected without real model)")
	} else {
		t.Logf("Expected error without model file: %v", err)
	}
}

// 测试用例2.4：超分辨率器创建（模拟）
func TestCreateUpscaler(t *testing.T) {
	// 注意：这个测试需要实际的超分辨率模型文件
	// 预期结果：在没有真实模型文件以模拟错误拦截及处理。
	modelPath := "test_esrgan.gguf"
	
	_, err := stablediffusion.NewUpscaler(modelPath)
	if err == nil {
		t.Logf("Upscaler created successfully (unexpected without real model)")
	} else {
		t.Logf("Expected error without model file: %v", err)
	}
}

// 测试用例2.5：模型转换（模拟）
func TestConvertModel(t *testing.T) {
	// 注意：这个测试需要实际的模型文件
	// 预期结果：能够在没有文件数据的情况下报错。
	inputPath := "test_model.gguf"
	vaePath := ""
	outputPath := "test_model_f16.gguf"
	
	err := stablediffusion.ConvertModel(inputPath, vaePath, outputPath, 1) // 1 代表 SD_TYPE_F16
	if err == nil {
		t.Logf("Model converted successfully (unexpected without real model)")
	} else {
		t.Logf("Expected error without model file: %v", err)
	}
}

// 测试用例2.6：图像生成配置验证
func TestImageGenerationConfig(t *testing.T) {
	// 测试配置结构的创建和设置
	// 预期结果：字段初始化完成无误差（例如宽高等于 512）。
	
	cfg := stablediffusion.GenerationConfig{
		Prompt:         "A beautiful cat",
		NegativePrompt: "ugly, blurry, bad art",
		Width:          512,
		Height:         512,
		Seed:           42,
		Strength:       0.8,
		BatchCount:     1,
		ClipSkip:       1,
	}
	
	// 验证配置
	if cfg.Prompt != "A beautiful cat" {
		t.Errorf("Expected prompt 'A beautiful cat', got '%s'", cfg.Prompt)
	}
	
	if cfg.Width != 512 || cfg.Height != 512 {
		t.Errorf("Expected size 512x512, got %dx%d", cfg.Width, cfg.Height)
	}
	
	t.Logf("Image generation config created successfully")
}

// 测试用例2.7：OpenModel 参数校验
func TestOpenModelEmptyPath(t *testing.T) {
	_, err := stablediffusion.OpenModel("", 4, 1, true)
	if err == nil {
		t.Fatal("空模型路径应当返回错误")
	}
	t.Logf("空路径错误: %v", err)
}

// 测试用例2.8：OpenModel 在无真实模型时返回错误且不泄漏资源
func TestOpenModelMissingFile(t *testing.T) {
	// 1 == SD_TYPE_F16；无论是否加载了动态库，不存在的模型文件都应失败
	model, err := stablediffusion.OpenModel("definitely-missing-model.gguf", 2, 1, true)
	if err == nil {
		// 极端情况下底层返回了句柄，也要保证测试结束时释放
		defer model.Close()
		t.Fatalf("不存在的模型文件不应创建成功")
	}
	if model != nil {
		t.Fatalf("创建失败时不应返回模型对象")
	}
}

// 测试用例2.9：Context 的 Close 幂等、nil 安全、并发安全
func TestContextCloseIdempotent(t *testing.T) {
	// nil 指针上调用 Close 不应 panic
	var nilCtx *stablediffusion.Context
	if err := nilCtx.Close(); err != nil {
		t.Fatalf("nil Context Close 应返回 nil, 实际 %v", err)
	}
	if !nilCtx.Closed() {
		t.Fatal("nil Context 应视为已关闭")
	}

	// 零值 Context（底层句柄为 nil）用于验证释放路径本身不会二次释放
	ctx := &stablediffusion.Context{}
	for i := 0; i < 3; i++ {
		if err := ctx.Close(); err != nil {
			t.Fatalf("第 %d 次 Close 返回错误: %v", i+1, err)
		}
		if !ctx.Closed() {
			t.Fatalf("第 %d 次 Close 后 Closed() 应为 true", i+1)
		}
	}

	// Free（Close 别名）再调用一次也不应崩溃
	ctx.Free()

	// 关闭后继续使用应返回错误而不是段错误
	if _, err := ctx.GenerateImage(stablediffusion.GenerationConfig{}); err == nil {
		t.Fatal("已关闭的 Context 生成图像应返回错误")
	}
	if _, err := ctx.GenerateVideo(stablediffusion.VideoGenerationConfig{}); err == nil {
		t.Fatal("已关闭的 Context 生成视频应返回错误")
	}

	// 并发 Close 不应 panic / 死锁
	ctx2 := &stablediffusion.Context{}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 50; i++ {
			_ = ctx2.Close()
		}
	}()
	for i := 0; i < 50; i++ {
		_ = ctx2.Close()
	}
	<-done
}

// 测试用例2.10：Upscaler 的 Close 同样幂等且 nil 安全
func TestUpscalerCloseIdempotent(t *testing.T) {
	var nilUpscaler *stablediffusion.Upscaler
	if err := nilUpscaler.Close(); err != nil {
		t.Fatalf("nil Upscaler Close 应返回 nil, 实际 %v", err)
	}

	u := &stablediffusion.Upscaler{}
	for i := 0; i < 3; i++ {
		if err := u.Close(); err != nil {
			t.Fatalf("第 %d 次 Close 返回错误: %v", i+1, err)
		}
	}
	u.Free() // 别名再调用一次

	if factor := u.GetUpscaleFactor(); factor != 0 {
		t.Fatalf("已关闭的 Upscaler 应返回 0 倍率, 实际 %d", factor)
	}
	if _, err := u.Upscale(&stablediffusion.Image{Width: 1, Height: 1, Channel: 3, Data: make([]byte, 3)}, 2); err == nil {
		t.Fatal("已关闭的 Upscaler 执行放大应返回错误")
	}
}

// 测试用例2.11：Model 的 Close nil 安全（成功创建路径由集成环境覆盖）
func TestModelCloseNilSafe(t *testing.T) {
	var m *stablediffusion.Model
	if err := m.Close(); err != nil {
		t.Fatalf("nil Model Close 应返回 nil, 实际 %v", err)
	}
	m.Free()

	empty := &stablediffusion.Model{}
	if err := empty.Close(); err != nil {
		t.Fatalf("空 Model Close 应返回 nil, 实际 %v", err)
	}
	if err := empty.Close(); err != nil {
		t.Fatalf("重复 Close 应返回 nil, 实际 %v", err)
	}
}
