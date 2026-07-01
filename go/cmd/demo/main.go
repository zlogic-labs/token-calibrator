// Demo: token-calibrator in action — loading from models.json.
//
// Shows how to initialise from a model snapshot file, then learn and
// export your own snapshot.
//
// Run: go run ./cmd/demo

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	calibrator "github.com/zlogic/token-calibrator/go"
)

func main() {
	fmt.Println("=== token-calibrator demo (Go) ===")

	// Resolve path to models/models.json relative to this source file
	_, srcFile, _, _ := runtime.Caller(0)
	modelsPath := filepath.Join(filepath.Dir(srcFile), "..", "..", "..", "models", "models.json")
	jsonData, err := os.ReadFile(modelsPath)
	if err != nil {
		// Fallback: try relative from cwd
		jsonData, err = os.ReadFile("models/models.json")
		if err != nil {
			panic("cannot find models/models.json: " + err.Error())
		}
	}

	english := "Hello, world! This is a test of the token calibrator."
	chinese := "你好世界，这是一个测试。"
	mixed := "Hello 你好 123 🎉"

	// 1. Load from models.json
	fmt.Println("── Load from models.json ──")
	cal := calibrator.NewTokenCalibratorFromModel("gpt-4o", jsonData)
	if cal == nil {
		cal = calibrator.NewTokenCalibratorFromModel("default", jsonData)
	}
	fmt.Println("Loaded model: gpt-4o")
	fmt.Printf("English estimate: %d tokens\n", cal.Estimate(english))
	fmt.Printf("Coefficients    : %v\n", cal.Coefficients())

	// 2. Compare with stateless priors
	fmt.Println("\n── Stateless priors (no calibration) ──")
	fmt.Printf("English  : %4d chars → ~%3d tokens\n",
		len(english), calibrator.EstimateTokensFromPriors(english))
	fmt.Printf("Chinese  : %4d chars → ~%3d tokens\n",
		len([]rune(chinese)), calibrator.EstimateTokensFromPriors(chinese))
	fmt.Printf("Mixed    : %4d chars → ~%3d tokens\n",
		len([]rune(mixed)), calibrator.EstimateTokensFromPriors(mixed))

	// 3. Feed observations
	fmt.Println("\n── Training with real observations ──")
	cal.Observe("Hello world", 3)
	cal.Observe("你好世界", 6)

	moreData := []struct {
		text   string
		tokens int
	}{
		{"short", 2},
		{"a bit longer english text here", 8},
		{"more english words for the model to learn from", 12},
		{"中文中文中文中文", 12},
		{"1234567890", 4},
	}
	for _, d := range moreData {
		cal.Observe(d.text, d.tokens)
	}
	fmt.Printf("After %d observations:\n", 2+len(moreData))
	fmt.Printf("English estimate: %d tokens\n", cal.Estimate(english))
	fmt.Printf("Chinese estimate: %d tokens\n", cal.Estimate(chinese))
	fmt.Printf("Coefficients    : %v\n", cal.Coefficients())

	// 4. Snapshot round-trip — load fresh from model and replay training
	fmt.Println("\n── Restored from fresh model + same training ──")
	restored := calibrator.NewTokenCalibratorFromModel("gpt-4o", jsonData)
	if restored == nil {
		restored = calibrator.NewTokenCalibratorFromModel("default", jsonData)
	}
	restored.Observe("Hello world", 3)
	restored.Observe("你好世界", 6)
	for _, d := range moreData {
		restored.Observe(d.text, d.tokens)
	}
	fmt.Printf("English estimate: %d (match = %v)\n",
		restored.Estimate(english), restored.Estimate(english) == cal.Estimate(english))

	// 5. Export your trained snapshot (to contribute back!)
	fmt.Println("\n── Your trained snapshot (ready to contribute!) ──")
	trained := cal.Snapshot()
	fmt.Printf("a: %v\n", trained.A)
	fmt.Printf("g: %v\n", trained.G)
	fmt.Printf("strength: %v\n", trained.Strength)

	fmt.Println("\n=== Demo complete ===")
}
