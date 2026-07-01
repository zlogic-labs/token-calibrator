package calibrator

import (
	"math"
	"testing"
)

func TestClassifyLatin(t *testing.T) {
	c := ClassifyTokenBuckets("Hello")
	if c[BucketLatin] != 5 {
		t.Errorf("expected 5 Latin, got %d", c[BucketLatin])
	}
	if c[BucketHan] != 0 {
		t.Errorf("expected 0 Han, got %d", c[BucketHan])
	}
	if c[BucketDigit] != 0 {
		t.Errorf("expected 0 Digit, got %d", c[BucketDigit])
	}
	if c[BucketOther] != 0 {
		t.Errorf("expected 0 Other, got %d", c[BucketOther])
	}
}

func TestClassifyChinese(t *testing.T) {
	c := ClassifyTokenBuckets("你好")
	if c[BucketHan] != 2 {
		t.Errorf("expected 2 Han, got %d", c[BucketHan])
	}
	if c[BucketLatin] != 0 {
		t.Errorf("expected 0 Latin, got %d", c[BucketLatin])
	}
}

func TestClassifyMixed(t *testing.T) {
	c := ClassifyTokenBuckets("Hello 世界 123 😊")
	// H e l l o + 3 spaces = 8 Latin; 世 界 = 2 Han; 1 2 3 = 3 Digit; 😊 = 4 bytes Other
	if c[BucketHan] != 2 {
		t.Errorf("expected 2 Han, got %d", c[BucketHan])
	}
	if c[BucketLatin] != 8 {
		t.Errorf("expected 8 Latin, got %d", c[BucketLatin])
	}
	if c[BucketDigit] != 3 {
		t.Errorf("expected 3 Digit, got %d", c[BucketDigit])
	}
	if c[BucketOther] != 4 {
		t.Errorf("expected 4 Other, got %d", c[BucketOther])
	}
}

func TestClassifyEmpty(t *testing.T) {
	c := ClassifyTokenBuckets("")
	if c != (BucketCounts{}) {
		t.Errorf("expected zero counts, got %v", c)
	}
}

func TestEstimateTokensFromPriors(t *testing.T) {
	n := EstimateTokensFromPriors("Hello world")
	if n != 3 {
		t.Errorf("expected 3, got %d", n)
	}
}

func TestEstimateTokensFromPriorsEmpty(t *testing.T) {
	n := EstimateTokensFromPriors("")
	if n != 0 {
		t.Errorf("expected 0, got %d", n)
	}
}

func TestCalibratorPriorEstimate(t *testing.T) {
	cal := NewTokenCalibrator(TokenCalibratorOptions{}, nil)
	n := cal.Estimate("Hello world")
	if n != 3 {
		t.Errorf("expected 3, got %d", n)
	}
}

func TestCalibratorAdapts(t *testing.T) {
	strength := 1.0
	opts := TokenCalibratorOptions{PriorStrength: &strength}
	cal := NewTokenCalibrator(opts, nil)
	for i := 0; i < 20; i++ {
		cal.Observe("Hello world", 4)
	}
	n := cal.Estimate("Hello world")
	if n != 4 {
		t.Errorf("expected 4 after training, got %d", n)
	}
}

func TestCalibratorSnapshotRoundtrip(t *testing.T) {
	cal := NewTokenCalibrator(TokenCalibratorOptions{}, nil)
	cal.Observe("test", 2)
	snap := cal.Snapshot()
	restored := NewTokenCalibrator(TokenCalibratorOptions{}, &snap)
	if restored.Estimate("test") != cal.Estimate("test") {
		t.Errorf("snapshot roundtrip mismatch: %d vs %d", restored.Estimate("test"), cal.Estimate("test"))
	}
}

func TestCoefficients(t *testing.T) {
	cal := NewTokenCalibrator(TokenCalibratorOptions{}, nil)
	coef := cal.Coefficients()
	for _, v := range coef {
		if math.IsInf(v, 0) || math.IsNaN(v) {
			t.Errorf("unexpected coefficient value: %v", v)
		}
	}
}

func TestSolveLinearSystem(t *testing.T) {
	// Identity matrix: A·x = b => x = b
	var a Matrix4
	for i := 0; i < N_BUCKETS; i++ {
		a[i][i] = 1.0
	}
	b := Coefficients{2.0, 3.0, 4.0, 5.0}
	x := solveLinearSystem(a, b)
	expected := Coefficients{2.0, 3.0, 4.0, 5.0}
	if x != expected {
		t.Errorf("expected %v, got %v", expected, x)
	}
}

func TestIsValidSnapshot(t *testing.T) {
	if isValidSnapshot(nil) {
		t.Error("nil snapshot should be invalid")
	}
	snap := TokenCalibratorSnapshot{
		A:        Matrix4{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},
		G:        Coefficients{1, 2, 3, 4},
		Strength: 1000,
	}
	if !isValidSnapshot(&snap) {
		t.Error("valid snapshot should be valid")
	}
}

func TestNewTokenCalibratorFromModel(t *testing.T) {
	jsonData := []byte(`{
		"models": {
			"test-model": {
				"a": [[1000,0,0,0],[0,1000,0,0],[0,0,1000,0],[0,0,0,1000]],
				"g": [1000,250,400,600],
				"strength": 1000
			}
		}
	}`)
	cal := NewTokenCalibratorFromModel("test-model", jsonData)
	if cal == nil {
		t.Fatal("expected non-nil calibrator")
	}
	n := cal.Estimate("Hello world")
	if n != 3 {
		t.Errorf("expected 3 from prior, got %d", n)
	}
}

func TestNewTokenCalibratorFromModelMissing(t *testing.T) {
	jsonData := []byte(`{"models": {}}`)
	cal := NewTokenCalibratorFromModel("nonexistent", jsonData)
	if cal != nil {
		t.Error("expected nil for missing model")
	}
}

func TestNewTokenCalibratorFromModelBadJSON(t *testing.T) {
	cal := NewTokenCalibratorFromModel("x", []byte("not json"))
	if cal != nil {
		t.Error("expected nil for bad JSON")
	}
}
