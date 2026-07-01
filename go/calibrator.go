// Package calibrator provides a calibrated, self-learning token estimator
// for live context-size display.
//
// It learns per-character token rates from real usage observations using
// online ridge regression. No bundled tokenizer is required — the estimator
// adapts to any model's actual tokenization behaviour.
//
// The four buckets are: Han, Latin, Digit, and Other (counted in UTF‑8 bytes).
// Priors act as a ridge penalty so the system is always well-conditioned and
// degrades gracefully to a heuristic when no data has been observed.
package calibrator

import (
	"encoding/json"
	"math"
)

// N_BUCKETS is the number of classification buckets.
const N_BUCKETS = 4

// Bucket index constants.
const (
	BucketHan   = 0
	BucketLatin = 1
	BucketDigit = 2
	BucketOther = 3
)

// BucketNames maps bucket indices to human-readable names.
var BucketNames = [N_BUCKETS]string{"han", "latin", "digit", "other"}

// Prior token-per-unit rates for each bucket.
// Han / Latin / Digit are per character; Other is per UTF‑8 byte.
var TokenBucketPriors = [N_BUCKETS]float64{1.0, 0.25, 0.4, 0.6}

const defaultPriorStrength = 1_000_000.0

// BucketCounts is a vector of counts for the four buckets.
type BucketCounts [N_BUCKETS]int

// Coefficients is a vector of per-bucket rates.
type Coefficients [N_BUCKETS]float64

// Matrix4 is a 4×4 matrix represented as an array of 4 rows.
type Matrix4 [N_BUCKETS][N_BUCKETS]float64

// utf8Len returns the number of UTF‑8 bytes needed to encode the given codepoint.
func utf8Len(cp rune) int {
	switch {
	case cp <= 0x7F:
		return 1
	case cp <= 0x7FF:
		return 2
	case cp <= 0xFFFF:
		return 3
	default:
		return 4
	}
}

// ClassifyTokenBuckets classifies a string into the four buckets.
// - Han: CJK ideographs (including extension planes)
// - Latin: printable ASCII, Latin‑1 supplement, Latin Extended‑A
// - Digit: ASCII digits 0–9
// - Other: everything else, counted in UTF‑8 bytes.
func ClassifyTokenBuckets(input string) BucketCounts {
	var counts BucketCounts
	for _, ch := range input {
		cp := int32(ch)
		switch {
		case cp >= 0x30 && cp <= 0x39:
			counts[BucketDigit]++
		case (cp >= 0x3400 && cp <= 0x4DBF) || // Extension A
			(cp >= 0x4E00 && cp <= 0x9FFF) || // Unified (URO)
			(cp >= 0xF900 && cp <= 0xFAFF) || // Compatibility Ideographs
			(cp >= 0x20000 && cp <= 0x2A6DF) || // Extension B
			(cp >= 0x2A700 && cp <= 0x2EBEF) || // Extensions C–F
			(cp >= 0x2F800 && cp <= 0x2FA1F) || // Compatibility Supplement
			(cp >= 0x30000 && cp <= 0x323AF): // Extensions G–H
			counts[BucketHan]++
		case cp >= 0x20 && cp <= 0x024F:
			counts[BucketLatin]++
		default:
			counts[BucketOther] += utf8Len(ch)
		}
	}
	return counts
}

// featureVector converts bucket counts to a float64 slice.
func featureVector(counts BucketCounts) [N_BUCKETS]float64 {
	return [N_BUCKETS]float64{
		float64(counts[0]),
		float64(counts[1]),
		float64(counts[2]),
		float64(counts[3]),
	}
}

// EstimateTokensFromPriors is a stateless estimate based solely on the priors.
func EstimateTokensFromPriors(input string) int {
	if input == "" {
		return 0
	}
	counts := ClassifyTokenBuckets(input)
	var sum float64
	for i, prior := range TokenBucketPriors {
		sum += prior * float64(counts[i])
	}
	n := int(math.Round(sum))
	if n < 1 {
		return 1
	}
	return n
}

// solveLinearSystem solves the 4×4 linear system A·x = b using Gauss‑Jordan
// elimination with partial pivoting.
func solveLinearSystem(a Matrix4, b Coefficients) Coefficients {
	n := N_BUCKETS
	// Build augmented matrix [A | b]
	var aug [N_BUCKETS][N_BUCKETS + 1]float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug[i][j] = a[i][j]
		}
		aug[i][n] = b[i]
	}

	for col := 0; col < n; col++ {
		pivot := col
		for r := col + 1; r < n; r++ {
			if math.Abs(aug[r][col]) > math.Abs(aug[pivot][col]) {
				pivot = r
			}
		}
		if math.Abs(aug[pivot][col]) < 1e-12 {
			continue
		}
		aug[col], aug[pivot] = aug[pivot], aug[col]

		diag := aug[col][col]
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			factor := aug[r][col] / diag
			if math.Abs(factor) < 1e-18 {
				continue
			}
			for c := col; c <= n; c++ {
				aug[r][c] -= factor * aug[col][c]
			}
		}
	}

	var x Coefficients
	for i := 0; i < n; i++ {
		diag := aug[i][i]
		if math.Abs(diag) < 1e-12 {
			x[i] = 0.0
		} else {
			x[i] = aug[i][n] / diag
		}
	}
	return x
}

// TokenCalibratorSnapshot represents the serializable state of a calibrator.
type TokenCalibratorSnapshot struct {
	A        Matrix4     `json:"a"`
	G        Coefficients `json:"g"`
	Strength float64     `json:"strength"`
}

// isValidSnapshot checks that a snapshot is structurally valid.
func isValidSnapshot(snap *TokenCalibratorSnapshot) bool {
	if math.IsInf(snap.Strength, 0) || math.IsNaN(snap.Strength) || snap.Strength <= 0 {
		return false
	}
	for _, v := range snap.G {
		if math.IsInf(v, 0) || math.IsNaN(v) {
			return false
		}
	}
	for _, row := range snap.A {
		for _, v := range row {
			if math.IsInf(v, 0) || math.IsNaN(v) {
				return false
			}
		}
	}
	return true
}

// TokenCalibratorOptions contains options for creating a calibrator.
type TokenCalibratorOptions struct {
	// PriorStrength controls the ridge penalty (default 1_000_000).
	PriorStrength *float64
	// Forgetting factor in (0,1]. 1.0 = no forgetting (default).
	Forgetting *float64
}

// TokenCalibrator is an online ridge-regression token estimator.
type TokenCalibrator struct {
	a        Matrix4
	g        Coefficients
	gamma    float64
	strength float64
	theta    *Coefficients
}

// NewTokenCalibrator creates a new calibrator.
func NewTokenCalibrator(opts TokenCalibratorOptions, snapshot *TokenCalibratorSnapshot) *TokenCalibrator {
	gamma := 1.0
	if opts.Forgetting != nil && *opts.Forgetting > 0 && *opts.Forgetting <= 1.0 {
		gamma = *opts.Forgetting
	}

	if snapshot != nil && isValidSnapshot(snapshot) {
		return &TokenCalibrator{
			a:        snapshot.A,
			g:        snapshot.G,
			gamma:    gamma,
			strength: snapshot.Strength,
		}
	}

	strength := defaultPriorStrength
	if opts.PriorStrength != nil && *opts.PriorStrength > 0 {
		strength = *opts.PriorStrength
	}
	cal := &TokenCalibrator{
		gamma:    gamma,
		strength: strength,
	}
	cal.seedPriors()
	return cal
}

func (c *TokenCalibrator) seedPriors() {
	for j := 0; j < N_BUCKETS; j++ {
		c.a[j][j] += c.strength
		c.g[j] += c.strength * TokenBucketPriors[j]
	}
}

func (c *TokenCalibrator) stripPriors() {
	for j := 0; j < N_BUCKETS; j++ {
		c.a[j][j] -= c.strength
		c.g[j] -= c.strength * TokenBucketPriors[j]
	}
}

func (c *TokenCalibrator) solve() Coefficients {
	if c.theta != nil {
		return *c.theta
	}
	theta := solveLinearSystem(c.a, c.g)
	c.theta = &theta
	return theta
}

// Estimate returns the estimated token count for the given input.
func (c *TokenCalibrator) Estimate(input string) int {
	if input == "" {
		return 0
	}
	counts := ClassifyTokenBuckets(input)
	x := featureVector(counts)
	theta := c.solve()
	var sum float64
	for j := 0; j < N_BUCKETS; j++ {
		sum += math.Max(0, theta[j]) * x[j]
	}
	n := int(math.Round(sum))
	if n < 1 {
		return 1
	}
	return n
}

// Observe incorporates one observed round: input whose real token count is `realTokens`.
func (c *TokenCalibrator) Observe(input string, realTokens int) {
	if realTokens <= 0 {
		return
	}
	counts := ClassifyTokenBuckets(input)
	x := featureVector(counts)
	// All-zero features carry no information
	allZero := true
	for _, v := range x {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return
	}

	c.stripPriors()

	if c.gamma < 1.0 {
		for i := 0; i < N_BUCKETS; i++ {
			c.g[i] *= c.gamma
			for j := 0; j < N_BUCKETS; j++ {
				c.a[i][j] *= c.gamma
			}
		}
	}

	rt := float64(realTokens)
	for i := 0; i < N_BUCKETS; i++ {
		c.g[i] += x[i] * rt
		for j := 0; j < N_BUCKETS; j++ {
			c.a[i][j] += x[i] * x[j]
		}
	}

	c.seedPriors()
	c.theta = nil
}

// Coefficients returns the current learned rates per bucket.
func (c *TokenCalibrator) Coefficients() Coefficients {
	return c.solve()
}

// Snapshot captures the current state for persistence.
func (c *TokenCalibrator) Snapshot() TokenCalibratorSnapshot {
	return TokenCalibratorSnapshot{
		A:        c.a,
		G:        c.g,
		Strength: c.strength,
	}
}

// ModelFileEntry is a single model entry in the models.json format.
type ModelFileEntry struct {
	A        Matrix4     `json:"a"`
	G        Coefficients `json:"g"`
	Strength float64     `json:"strength"`
}

// NewTokenCalibratorFromModel reads a models.json-format byte slice and
// returns a calibrator seeded with the snapshot for the given model name.
// If the model is not found or the JSON is malformed, it returns a nil
// calibrator — callers can fall back to NewTokenCalibrator.
func NewTokenCalibratorFromModel(name string, jsonData []byte) *TokenCalibrator {
	var file struct {
		Models map[string]ModelFileEntry `json:"models"`
	}
	if err := json.Unmarshal(jsonData, &file); err != nil {
		return nil
	}
	entry, ok := file.Models[name]
	if !ok {
		return nil
	}
	snap := TokenCalibratorSnapshot{
		A:        entry.A,
		G:        entry.G,
		Strength: entry.Strength,
	}
	if !isValidSnapshot(&snap) {
		return nil
	}
	return NewTokenCalibrator(TokenCalibratorOptions{}, &snap)
}
