package calibrator

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// classify_token_buckets
// ---------------------------------------------------------------------------

func TestClassifyLatin(t *testing.T) {
	c := ClassifyTokenBuckets("Hello")
	if c["han"] != 0 || c["latin"] != 5 || c["digit"] != 0 || c["hangul"] != 0 || c["cyrillic"] != 0 || c["emoji"] != 0 || c["other"] != 0 {
		t.Errorf("unexpected classification: %v", c)
	}
}

func TestClassifyChinese(t *testing.T) {
	c := ClassifyTokenBuckets("你好")
	if c["han"] != 2 || c["latin"] != 0 || c["other"] != 0 {
		t.Errorf("unexpected classification: %v", c)
	}
}

func TestClassifyEmpty(t *testing.T) {
	c := ClassifyTokenBuckets("")
	for _, b := range TOKEN_BUCKETS {
		if c[b] != 0 {
			t.Errorf("expected 0 for %s, got %f", b, c[b])
		}
	}
}

func TestClassifyKorean(t *testing.T) {
	c := ClassifyTokenBuckets("안녕하세요")
	if c["hangul"] != 5 {
		t.Errorf("expected 5 hangul, got %f", c["hangul"])
	}
}

func TestClassifyCyrillic(t *testing.T) {
	c := ClassifyTokenBuckets("Привет")
	if c["cyrillic"] != 6 {
		t.Errorf("expected 6 cyrillic, got %f", c["cyrillic"])
	}
}

func TestClassifyEmoji(t *testing.T) {
	c := ClassifyTokenBuckets("🚀🎉")
	if c["emoji"] != 2 {
		t.Errorf("expected 2 emoji, got %f", c["emoji"])
	}
}

// ---------------------------------------------------------------------------
// feature_vector
// ---------------------------------------------------------------------------

func TestFeatureVectorOrder(t *testing.T) {
	counts := TokenRates{"han": 2, "latin": 5, "digit": 3, "hangul": 0, "cyrillic": 0, "emoji": 1, "other": 4}
	fv := FeatureVector(counts)
	expected := [7]float64{2, 5, 3, 0, 0, 1, 4}
	if fv != expected {
		t.Errorf("expected %v, got %v", expected, fv)
	}
}

// ---------------------------------------------------------------------------
// estimate_tokens
// ---------------------------------------------------------------------------

func TestEstimatePriorHelloWorld(t *testing.T) {
	priors := TOKEN_BUCKET_PRIORS()
	if n := EstimateTokens("Hello world", priors); n != 3 {
		t.Errorf("expected 3, got %d", n)
	}
}

func TestEstimateEmpty(t *testing.T) {
	priors := TOKEN_BUCKET_PRIORS()
	if n := EstimateTokens("", priors); n != 0 {
		t.Errorf("expected 0, got %d", n)
	}
}

func TestEstimateFloorAt1(t *testing.T) {
	priors := TOKEN_BUCKET_PRIORS()
	if n := EstimateTokens("a", priors); n != 1 {
		t.Errorf("expected 1, got %d", n)
	}
}

// ---------------------------------------------------------------------------
// empty_accumulator / is_valid_accumulator
// ---------------------------------------------------------------------------

func TestEmptyAccumulatorIsValid(t *testing.T) {
	acc := EmptyAccumulator()
	if !IsValidAccumulator(&acc) {
		t.Error("expected valid accumulator")
	}
}

func TestNanIsInvalid(t *testing.T) {
	acc := EmptyAccumulator()
	acc.A[0][0] = math.NaN()
	if IsValidAccumulator(&acc) {
		t.Error("expected invalid accumulator")
	}
}

func TestInfIsInvalid(t *testing.T) {
	acc := EmptyAccumulator()
	acc.G[0] = math.Inf(1)
	if IsValidAccumulator(&acc) {
		t.Error("expected invalid accumulator")
	}
}

// ---------------------------------------------------------------------------
// accumulate / rates_from_accumulator
// ---------------------------------------------------------------------------

func TestNonPositiveTokensIgnored(t *testing.T) {
	acc := EmptyAccumulator()
	acc2 := Accumulate(acc, "Hello", 0, nil)
	if acc != acc2 {
		t.Error("expected same accumulator")
	}
}

func TestAccumulateAndSolve(t *testing.T) {
	acc := EmptyAccumulator()
	for i := 0; i < 20; i++ {
		acc = Accumulate(acc, "Hello world", 4, nil)
	}
	strength := 1.0
	rates := RatesFromAccumulator(acc, &strength, nil)
	latin := rates["latin"]
	expected := 4.0 / 11.0
	if math.Abs(latin-expected) > 0.1 {
		t.Errorf("latin rate %f, expected ~%f", latin, expected)
	}
}

// ---------------------------------------------------------------------------
// solve_linear_system
// ---------------------------------------------------------------------------

func TestSolveIdentity(t *testing.T) {
	var a [N_BUCKETS][N_BUCKETS]float64
	for i := 0; i < N_BUCKETS; i++ {
		a[i][i] = 1.0
	}
	b := [N_BUCKETS]float64{2, 3, 4, 5, 6, 7, 8}
	x := SolveLinearSystem(a, b)
	expected := [N_BUCKETS]float64{2, 3, 4, 5, 6, 7, 8}
	if x != expected {
		t.Errorf("expected %v, got %v", expected, x)
	}
}

// ---------------------------------------------------------------------------
// TokenCalibrator
// ---------------------------------------------------------------------------

func TestCalibratorStartsWithPriorRates(t *testing.T) {
	strength := 1.0
	cal := NewTokenCalibrator(TokenCalibratorOptions{PriorStrength: &strength})
	r := cal.Rates()
	if math.Abs(r["han"]-1.0) > 1e-9 {
		t.Errorf("han rate %f, expected 1.0", r["han"])
	}
	if math.Abs(r["latin"]-0.25) > 1e-9 {
		t.Errorf("latin rate %f, expected 0.25", r["latin"])
	}
}

func TestCalibratorAdapts(t *testing.T) {
	strength := 1.0
	cal := NewTokenCalibrator(TokenCalibratorOptions{PriorStrength: &strength})
	for i := 0; i < 20; i++ {
		cal.Observe("Hello world", 4)
	}
	r := cal.Rates()
	latin := r["latin"]
	if math.Abs(latin-4.0/11.0) > 0.1 {
		t.Errorf("latin rate %f, expected ~%f", latin, 4.0/11.0)
	}
}

func TestCalibratorEstimatePrior(t *testing.T) {
	strength := 1.0
	cal := NewTokenCalibrator(TokenCalibratorOptions{PriorStrength: &strength})
	if n := cal.Estimate("Hello world"); n != 3 {
		t.Errorf("expected 3, got %d", n)
	}
}

func TestCalibratorEstimateAfterTraining(t *testing.T) {
	strength := 1.0
	cal := NewTokenCalibrator(TokenCalibratorOptions{PriorStrength: &strength})
	for i := 0; i < 20; i++ {
		cal.Observe("Hello world", 4)
	}
	if n := cal.Estimate("Hello world"); n != 4 {
		t.Errorf("expected 4, got %d", n)
	}
}

func TestCalibratorRoundTripMatrix(t *testing.T) {
	cal := NewTokenCalibrator(TokenCalibratorOptions{})
	cal.Observe("test", 2)
	matrix := cal.ToMatrix()
	restored := NewTokenCalibrator(TokenCalibratorOptions{Accumulator: &matrix})
	if restored.Estimate("test") != cal.Estimate("test") {
		t.Error("round-trip estimate mismatch")
	}
}

// ---------------------------------------------------------------------------
// TokenEstimator
// ---------------------------------------------------------------------------

func TestEstimatorUnknownModel(t *testing.T) {
	est := NewTokenEstimator(nil, TokenEstimatorOptions{})
	if _, ok := BUILTIN_TOKEN_RATES["gpt-4o"]; ok {
		n := est.Estimate("gpt-4o", "Hello world")
		if n <= 0 {
			t.Errorf("expected positive, got %d", n)
		}
	}
}

func TestEstimatorFromRegisteredModel(t *testing.T) {
	acc := EmptyAccumulator()
	for i := 0; i < 20; i++ {
		acc = Accumulate(acc, "Hello world", 4, nil)
	}
	strength := 1.0
	matrices := map[string]TokenAccumulator{"test-model": acc}
	est := NewTokenEstimator(matrices, TokenEstimatorOptions{PriorStrength: &strength})
	if n := est.Estimate("test-model", "Hello world"); n != 4 {
		t.Errorf("expected 4, got %d", n)
	}
}

func TestEstimatorEmptyInput(t *testing.T) {
	matrices := map[string]TokenAccumulator{"m": EmptyAccumulator()}
	est := NewTokenEstimator(matrices, TokenEstimatorOptions{})
	if n := est.Estimate("m", ""); n != 0 {
		t.Errorf("expected 0, got %d", n)
	}
}

func TestEstimatorHasModel(t *testing.T) {
	matrices := map[string]TokenAccumulator{"m": EmptyAccumulator()}
	est := NewTokenEstimator(matrices, TokenEstimatorOptions{})
	if !est.Has("m") {
		t.Error("expected Has(m) = true")
	}
	if est.Has("nonexistent") {
		t.Error("expected Has(nonexistent) = false")
	}
}

func TestEstimatorRatesFallback(t *testing.T) {
	est := NewTokenEstimator(nil, TokenEstimatorOptions{})
	r := est.Rates("nonexistent-model-xyz")
	priors := TOKEN_BUCKET_PRIORS()
	for k, v := range priors {
		if r[k] != v {
			t.Errorf("expected %s = %f, got %f", k, v, r[k])
		}
	}
}
