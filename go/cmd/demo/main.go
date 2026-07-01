// Demo: token-calibrator in action.
//
// Showcases prior estimates, learning from observations, and how the model
// adapts its coefficients.
//
// Run: go run ./cmd/demo

package main

import (
	"fmt"

	"github.com/zlogic/token-calibrator/go/calibrator"
)

func main() {
	fmt.Println("=== token-calibrator demo (Go) ===\n")

	// 1. Stateless prior estimate
	english := "Hello, world! This is a test of the token calibrator."
	chinese := "你好世界，这是一个测试。"
	mixed := "Hello 你好 123 🎉"

	fmt.Println("── Priors (no learning yet) ──")
	fmt.Printf("English  : %4d chars → ~%3d tokens\n",
		len(english), calibrator.EstimateTokensFromPriors(english))
	fmt.Printf("Chinese  : %4d chars → ~%3d tokens\n",
		len([]rune(chinese)), calibrator.EstimateTokensFromPriors(chinese))
	fmt.Printf("Mixed    : %4d chars → ~%3d tokens\n",
		len([]rune(mixed)), calibrator.EstimateTokensFromPriors(mixed))

	// 2. Create a calibrator and learn
	cal := calibrator.NewTokenCalibrator(calibrator.TokenCalibratorOptions{}, nil)
	fmt.Println("\n── Before observing ──")
	fmt.Printf("English estimate: %d tokens\n", cal.Estimate(english))
	fmt.Printf("Coefficients    : %v\n", cal.Coefficients())

	// 3. Feed two observations
	cal.Observe("Hello world", 3)
	cal.Observe("你好世界", 6)

	fmt.Println("\n── After observing 2 rounds ──")
	fmt.Printf("English estimate: %d tokens (was %d)\n",
		cal.Estimate(english), calibrator.EstimateTokensFromPriors(english))
	fmt.Printf("Coefficients    : %v\n", cal.Coefficients())

	// 4. Feed more data
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
	fmt.Printf("\n── After %d more observations ──\n", len(moreData))
	fmt.Printf("English estimate: %d tokens\n", cal.Estimate(english))
	fmt.Printf("Chinese estimate: %d tokens\n", cal.Estimate(chinese))
	fmt.Printf("Coefficients    : %v\n", cal.Coefficients())

	// 5. Snapshot round-trip
	snap := cal.Snapshot()
	restored := calibrator.NewTokenCalibrator(calibrator.TokenCalibratorOptions{}, &snap)
	fmt.Printf("\n── Restored from snapshot ──\n")
	fmt.Printf("English estimate: %d (match = %v)\n",
		restored.Estimate(english), restored.Estimate(english) == cal.Estimate(english))

	fmt.Println("\n=== Demo complete ===")
}
