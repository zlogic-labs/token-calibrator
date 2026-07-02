// Smoke test: github.com/zlogic-libs/token-calibrator/go
// Run: go run main.go
package main

import (
	"fmt"
	"sort"

	calibrator "github.com/zlogic-libs/token-calibrator/go"
)

func main() {
	// 1. Basic classify
	c := calibrator.ClassifyTokenBuckets("Hello 世界 123")
	fmt.Printf("[GO] classifyTokenBuckets: %v\n", c)

	// 2. Prior estimate
	priors := calibrator.TOKEN_BUCKET_PRIORS()
	t1 := calibrator.EstimateTokens("Hello world", priors)
	fmt.Printf("[GO] estimate (prior): %d (expected 3)\n", t1)

	// 3. Calibrate + estimate
	strength := 1000.0
	cal := calibrator.NewTokenCalibrator(calibrator.TokenCalibratorOptions{
		PriorStrength: &strength,
	})
	observations := []struct {
		text   string
		tokens float64
	}{
		{"Hello world", 3},
		{"你好世界", 6},
		{"12345", 5},
	}
	for _, obs := range observations {
		cal.Observe(obs.text, obs.tokens)
	}
	t2 := cal.Estimate("Hello world")
	fmt.Printf("[GO] calibrate + estimate: %d (expected ~3)\n", t2)

	// 4. Estimator with built-in models
	est := calibrator.NewTokenEstimator(nil, calibrator.TokenEstimatorOptions{})
	modelNames := make([]string, 0, 3)
	for name := range calibrator.BUILTIN_TOKEN_RATES {
		modelNames = append(modelNames, name)
		if len(modelNames) >= 3 {
			break
		}
	}
	sort.Strings(modelNames)
	for _, model := range modelNames {
		t := est.Estimate(model, "Hello world")
		fmt.Printf("[GO] %s: %d tokens\n", model, t)
	}

	// 5. Estimator with custom matrix
	matrix := cal.ToMatrix()
	matrices := map[string]calibrator.TokenAccumulator{"my-model": matrix}
	est2 := calibrator.NewTokenEstimator(matrices, calibrator.TokenEstimatorOptions{})
	fmt.Printf("[GO] custom model estimate: %d\n", est2.Estimate("my-model", "Hello world"))

	fmt.Println("[GO] All smoke tests passed!")
}
