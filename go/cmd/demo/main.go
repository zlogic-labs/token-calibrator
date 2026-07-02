package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/zlogic/token-calibrator/go"
)

func main() {
	args := os.Args[1:]
	mode := "estimate"
	if len(args) > 0 {
		mode = args[0]
	}

	fmt.Printf("token-calibrator demo (mode: %s)\n\n", mode)

	switch mode {
	case "calibrate":
		cmdCalibrate()
	default:
		snapshotPath := ""
		if len(args) > 1 {
			snapshotPath = args[1]
		}
		cmdEstimate(snapshotPath)
	}
}

// ────────────────────── Calibrate mode ──────────────────────

func cmdCalibrate() {
	fmt.Println("=== Calibrate mode ===")
	fmt.Println()

	strength := 1000.0
	cal := calibrator.NewTokenCalibrator(calibrator.TokenCalibratorOptions{
		PriorStrength: &strength,
	})

	type observation struct {
		text   string
		tokens float64
	}

	observations := []observation{
		{"Hello world", 3},
		{"The quick brown fox jumps over the lazy dog", 10},
		{"你好，世界", 6},
		{"안녕하세요", 8},
		{"Привет мир", 5},
		{"123 456 7890", 6},
		{"🚀 Token estimation is amazing! 🎉", 12},
		{"Mixed 你好 Hello 123 😊", 9},
	}

	for _, obs := range observations {
		counts := calibrator.ClassifyTokenBuckets(obs.text)
		fmt.Printf("  observe: %v  →  %.0f tokens  (%q)\n", counts, obs.tokens, obs.text)
		cal.Observe(obs.text, obs.tokens)
	}

	rates := cal.Rates()
	fmt.Println()
	fmt.Println("Learned per-bucket rates:")
	keys := make([]string, 0, len(rates))
	for k := range rates {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Printf("  %-10s %.4f\n", k, rates[k])
	}

	fmt.Println()
	fmt.Println("Estimates after calibration:")
	for _, text := range []string{"Hello world", "你好", "Mixed 你好 123 😊"} {
		fmt.Printf("  %-30q → %d tokens\n", text, cal.Estimate(text))
	}

	matrix := cal.ToMatrix()
	snapshot := map[string]any{
		"models": map[string]calibrator.TokenAccumulator{
			"demo-model": matrix,
		},
	}
	data, _ := json.MarshalIndent(snapshot, "", "  ")
	os.WriteFile("calibrated-snapshot.json", data, 0644)
	fmt.Println()
	fmt.Println("Exported accumulator to calibrated-snapshot.json")
	fmt.Println()
}

// ───────────────────── Estimate mode ─────────────────────

func cmdEstimate(snapshotPath string) {
	fmt.Println("=== Estimate mode ===")
	fmt.Println()

	matrices := make(map[string]calibrator.TokenAccumulator)

	if snapshotPath != "" {
		if _, err := os.Stat(snapshotPath); err == nil {
			data, _ := os.ReadFile(snapshotPath)
			var raw map[string]any
			json.Unmarshal(data, &raw)
			if models, ok := raw["models"].(map[string]any); ok {
				for name, val := range models {
					bytes, _ := json.Marshal(val)
					var acc calibrator.TokenAccumulator
					json.Unmarshal(bytes, &acc)
					matrices[name] = acc
				}
			}
			fmt.Printf("Loaded %d model(s) from %s\n", len(matrices), snapshotPath)
			fmt.Println()
		} else {
			fmt.Printf("File not found: %s, using built-in models\n", snapshotPath)
			fmt.Println()
		}
	} else {
		fmt.Println("Using built-in default models (no snapshot file provided)")
		fmt.Println()
	}

	est := calibrator.NewTokenEstimator(matrices, calibrator.TokenEstimatorOptions{})

	testTexts := []string{
		"Hello world",
		"The quick brown fox jumps over the lazy dog",
		"你好，世界",
		"안녕하세요",
		"Привет мир",
		"123 456 7890",
		"🚀 Token estimation is amazing! 🎉",
	}

	// A few model names
	modelNames := make([]string, 0, 6)
	for name := range calibrator.BUILTIN_TOKEN_RATES {
		modelNames = append(modelNames, name)
		if len(modelNames) >= 5 {
			break
		}
	}
	modelNames = append(modelNames, "demo-model")

	for _, model := range modelNames {
		fmt.Printf("── %s ──\n", model)
		if est.Has(model) {
			fmt.Println("  (user-calibrated data loaded)")
		}
		for _, text := range testTexts {
			t := est.Estimate(model, text)
			fmt.Printf("  %-50q → %d tokens\n", text, t)
		}
		fmt.Println()
	}

	fmt.Println("── unknown-model (falls back to prior) ──")
	for _, text := range []string{"Hello", "你好"} {
		t := est.Estimate("unknown-model", text)
		fmt.Printf("  %-50q → %d tokens\n", text, t)
	}
}
