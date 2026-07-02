// Package calibrator provides a calibrated, self-learning token estimator
// for live context-size display.
//
// It learns per-character token rates from real usage observations using
// online ridge regression. No bundled tokenizer is required — the estimator
// adapts to any model's actual tokenization behaviour.
//
// The seven buckets are: Han, Latin, Digit, Hangul, Cyrillic, Emoji, and
// Other (counted in UTF-8 bytes).
package calibrator

// TOKEN_BUCKETS lists all bucket names in order.
var TOKEN_BUCKETS = []string{
	"han", "latin", "digit", "hangul", "cyrillic", "emoji", "other",
}

// N_BUCKETS is the dimension of the accumulator / feature vector.
const N_BUCKETS = 7

// TokenRates maps bucket names to their learned token rates.
type TokenRates map[string]float64

// TOKEN_BUCKET_PRIORS returns a new map with the default prior rates.
func TOKEN_BUCKET_PRIORS() TokenRates {
	return TokenRates{
		"han":      1.0,
		"latin":    0.25,
		"digit":    0.4,
		"hangul":   1.2,
		"cyrillic": 0.5,
		"emoji":    1.5,
		"other":    0.6,
	}
}

func utf8Len(cp rune) int {
	if cp <= 0x7F {
		return 1
	} else if cp <= 0x7FF {
		return 2
	} else if cp <= 0xFFFF {
		return 3
	}
	return 4
}

// isHan returns true if cp is a CJK ideograph.
func isHan(cp rune) bool {
	return (cp >= 0x3400 && cp <= 0x4DBF) ||
		(cp >= 0x4E00 && cp <= 0x9FFF) ||
		(cp >= 0xF900 && cp <= 0xFAFF) ||
		(cp >= 0x20000 && cp <= 0x2A6DF) ||
		(cp >= 0x2A700 && cp <= 0x2EBEF) ||
		(cp >= 0x2F800 && cp <= 0x2FA1F) ||
		(cp >= 0x30000 && cp <= 0x323AF)
}

func isHangul(cp rune) bool {
	return (cp >= 0xAC00 && cp <= 0xD7A3) ||
		(cp >= 0x1100 && cp <= 0x11FF) ||
		(cp >= 0x3130 && cp <= 0x318F)
}

func isCyrillic(cp rune) bool {
	return (cp >= 0x0400 && cp <= 0x04FF) ||
		(cp >= 0x0500 && cp <= 0x052F)
}

func isEmoji(cp rune) bool {
	return (cp >= 0x1F000 && cp <= 0x1FAFF) ||
		(cp >= 0x2600 && cp <= 0x26FF) ||
		(cp >= 0x2700 && cp <= 0x27BF)
}

// ClassifyTokenBuckets classifies a string into per-bucket counts.
// Every bucket is a CHARACTER count except `other`, which is a UTF-8 BYTE count.
func ClassifyTokenBuckets(input string) TokenRates {
	counts := TokenRates{
		"han":      0,
		"latin":    0,
		"digit":    0,
		"hangul":   0,
		"cyrillic": 0,
		"emoji":    0,
		"other":    0,
	}
	if input == "" {
		return counts
	}

	for _, ch := range input {
		switch {
		case ch >= 0x30 && ch <= 0x39:
			counts["digit"]++
		case isHan(ch):
			counts["han"]++
		case ch >= 0x20 && ch <= 0x024F:
			counts["latin"]++
		case isHangul(ch):
			counts["hangul"]++
		case isCyrillic(ch):
			counts["cyrillic"]++
		case isEmoji(ch):
			counts["emoji"]++
		default:
			counts["other"] += float64(utf8Len(ch))
		}
	}
	return counts
}

// FeatureVector returns an ordered feature vector in TOKEN_BUCKETS order.
func FeatureVector(counts TokenRates) [N_BUCKETS]float64 {
	var v [N_BUCKETS]float64
	for i, b := range TOKEN_BUCKETS {
		v[i] = counts[b]
	}
	return v
}

// EstimateTokens estimates the token count for input using the given rates.
func EstimateTokens(input string, rates TokenRates) int {
	if input == "" {
		return 0
	}
	counts := ClassifyTokenBuckets(input)
	var sum float64
	for _, b := range TOKEN_BUCKETS {
		rate := rates[b]
		if rate < 0 {
			rate = 0
		}
		sum += rate * counts[b]
	}
	result := int(sum + 0.5) // round
	if result < 1 {
		return 1
	}
	return result
}
