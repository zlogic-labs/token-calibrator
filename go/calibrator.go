package calibrator

// TokenCalibratorOptions holds optional parameters for TokenCalibrator.
type TokenCalibratorOptions struct {
	Forgetting    *float64
	PriorStrength *float64
	Prior         *TokenRates
	Accumulator   *TokenAccumulator
}

// TokenCalibrator is a stateful writer around ONE model's accumulator.
type TokenCalibrator struct {
	acc           TokenAccumulator
	forgetting    *float64
	priorStrength *float64
	prior         *TokenRates
}

// NewTokenCalibrator creates a new TokenCalibrator.
func NewTokenCalibrator(opts TokenCalibratorOptions) *TokenCalibrator {
	cal := &TokenCalibrator{
		forgetting:    opts.Forgetting,
		priorStrength: opts.PriorStrength,
		prior:         opts.Prior,
		acc:           EmptyAccumulator(),
	}
	if opts.Accumulator != nil && IsValidAccumulator(opts.Accumulator) {
		cal.acc = *opts.Accumulator
	}
	return cal
}

// Observe folds in one observed round.
func (c *TokenCalibrator) Observe(input string, realTokens float64) {
	c.acc = Accumulate(c.acc, input, realTokens, c.forgetting)
}

// Rates returns the current solved per-bucket rates.
func (c *TokenCalibrator) Rates() TokenRates {
	return RatesFromAccumulator(c.acc, c.priorStrength, c.prior)
}

// Estimate directly from this calibrator's current rates.
func (c *TokenCalibrator) Estimate(input string) int {
	return EstimateTokens(input, c.Rates())
}

// ToMatrix returns the raw data-only accumulator for persistence.
func (c *TokenCalibrator) ToMatrix() TokenAccumulator {
	return c.acc
}
