package calibrator

// TokenAccumulator is a data-only least-squares accumulator.
// a = Σ xxT (N×N), g = Σ x·y (N). The ridge prior is NOT baked in.
type TokenAccumulator struct {
	A [N_BUCKETS][N_BUCKETS]float64 `json:"a"`
	G [N_BUCKETS]float64            `json:"g"`
}

const defaultPriorStrength = 1_000_000.0

// EmptyAccumulator returns a fresh, empty data-only accumulator.
func EmptyAccumulator() TokenAccumulator {
	return TokenAccumulator{}
}

// Accumulate folds one observed round into the accumulator and returns a NEW one.
// All-zero features or non-positive tokens are ignored.
func Accumulate(acc TokenAccumulator, input string, realTokens float64, forgetting *float64) TokenAccumulator {
	if !(realTokens > 0) {
		return acc
	}
	x := FeatureVector(ClassifyTokenBuckets(input))
	allZero := true
	for _, v := range x {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return acc
	}

	gamma := 1.0
	if forgetting != nil && *forgetting > 0 {
		if *forgetting < 1.0 {
			gamma = *forgetting
		}
	}

	a := acc.A
	g := acc.G

	for i := 0; i < N_BUCKETS; i++ {
		if gamma < 1.0 {
			g[i] *= gamma
			for j := 0; j < N_BUCKETS; j++ {
				a[i][j] *= gamma
			}
		}
		g[i] += x[i] * realTokens
		for j := 0; j < N_BUCKETS; j++ {
			a[i][j] += x[i] * x[j]
		}
	}

	return TokenAccumulator{A: a, G: g}
}

// RatesFromAccumulator solves the ridge-regularized least squares.
func RatesFromAccumulator(acc TokenAccumulator, priorStrength *float64, prior *TokenRates) TokenRates {
	lambda := defaultPriorStrength
	if priorStrength != nil && *priorStrength > 0 {
		lambda = *priorStrength
	}
	priorMap := TOKEN_BUCKET_PRIORS()
	if prior != nil {
		priorMap = *prior
	}

	a := acc.A
	g := acc.G

	for j, bucket := range TOKEN_BUCKETS {
		a[j][j] += lambda
		g[j] += lambda * priorMap[bucket]
	}

	theta := solveLinearSystem(&a, &g)
	rates := make(TokenRates)
	for j, bucket := range TOKEN_BUCKETS {
		rates[bucket] = theta[j]
	}
	return rates
}

// DeriveRates derives per-model rates from a map of accumulators.
func DeriveRates(matrices map[string]TokenAccumulator, priorStrength *float64, prior *TokenRates) map[string]TokenRates {
	out := make(map[string]TokenRates)
	for model, acc := range matrices {
		if IsValidAccumulator(&acc) {
			out[model] = RatesFromAccumulator(acc, priorStrength, prior)
		}
	}
	return out
}

// IsValidAccumulator validates an accumulator structurally.
func IsValidAccumulator(acc *TokenAccumulator) bool {
	for _, row := range acc.A {
		for _, v := range row {
			if !isFinite(v) {
				return false
			}
		}
	}
	for _, v := range acc.G {
		if !isFinite(v) {
			return false
		}
	}
	return true
}

func isFinite(v float64) bool {
	return v == v && v <= 1e308 && v >= -1e308
}

// SolveLinearSystem performs Gauss-Jordan elimination with partial pivoting.
func SolveLinearSystem(a [N_BUCKETS][N_BUCKETS]float64, b [N_BUCKETS]float64) [N_BUCKETS]float64 {
	return solveLinearSystem(&a, &b)
}

func solveLinearSystem(a *[N_BUCKETS][N_BUCKETS]float64, b *[N_BUCKETS]float64) [N_BUCKETS]float64 {
	n := N_BUCKETS
	// augmented matrix [A | b]
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n+1)
		for j := 0; j < n; j++ {
			m[i][j] = a[i][j]
		}
		m[i][n] = b[i]
	}

	for col := 0; col < n; col++ {
		pivot := col
		for r := col + 1; r < n; r++ {
			if abs(m[r][col]) > abs(m[pivot][col]) {
				pivot = r
			}
		}
		if abs(m[pivot][col]) < 1e-12 {
			continue
		}

		m[col], m[pivot] = m[pivot], m[col]
		diag := m[col][col]
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			factor := m[r][col] / diag
			if factor == 0 {
				continue
			}
			for c := col; c <= n; c++ {
				m[r][c] -= factor * m[col][c]
			}
		}
	}

	var x [N_BUCKETS]float64
	for i := 0; i < n; i++ {
		diag := m[i][i]
		if abs(diag) < 1e-12 {
			x[i] = 0
		} else {
			x[i] = m[i][n] / diag
		}
	}
	return x
}

func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}
