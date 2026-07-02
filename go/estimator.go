package calibrator

// TokenEstimatorOptions holds optional parameters for TokenEstimator.
type TokenEstimatorOptions struct {
	PriorStrength *float64
	Baseline      *map[string]TokenRates
	Prior         *TokenRates
}

// TokenEstimator is a read-only, per-model estimator.
type TokenEstimator struct {
	ratesByModel map[string]TokenRates
	baseline     map[string]TokenRates
	prior        TokenRates
}

// NewTokenEstimator creates a new TokenEstimator.
func NewTokenEstimator(matrices map[string]TokenAccumulator, opts TokenEstimatorOptions) *TokenEstimator {
	est := &TokenEstimator{
		ratesByModel: make(map[string]TokenRates),
		baseline:     BUILTIN_TOKEN_RATES,
		prior:        TOKEN_BUCKET_PRIORS(),
	}

	if opts.Baseline != nil {
		est.baseline = *opts.Baseline
	}
	if opts.Prior != nil {
		est.prior = *opts.Prior
	}

	for model, acc := range matrices {
		if !IsValidAccumulator(&acc) {
			continue
		}
		modelPrior, ok := est.baseline[model]
		if !ok {
			modelPrior = est.prior
		}
		est.ratesByModel[model] = RatesFromAccumulator(acc, opts.PriorStrength, &modelPrior)
	}

	return est
}

// Estimate estimates tokens for input under model's effective rates.
func (e *TokenEstimator) Estimate(model, input string) int {
	return EstimateTokens(input, e.Rates(model))
}

// Rates returns the effective rates for model.
func (e *TokenEstimator) Rates(model string) TokenRates {
	if r, ok := e.ratesByModel[model]; ok {
		return r
	}
	if r, ok := e.baseline[model]; ok {
		return r
	}
	return e.prior
}

// Has returns whether this model has user-calibrated rates.
func (e *TokenEstimator) Has(model string) bool {
	_, ok := e.ratesByModel[model]
	return ok
}
