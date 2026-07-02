package calibrator

func init() {
	// Initialize BUILTIN_TOKEN_RATES at package init.
	initBuiltinRates()
}

var BUILTIN_TOKEN_RATES map[string]TokenRates

func initBuiltinRates() {
	BUILTIN_TOKEN_RATES = make(map[string]TokenRates)

	deepseek := TokenRates{"han": 0.6896, "latin": 0.2521, "digit": 0.838, "hangul": 0.8542, "cyrillic": 0.2955, "emoji": 2.0485, "other": 0.1799}
	gpt3 := TokenRates{"han": 1.2812, "latin": 0.2458, "digit": 0.7706, "hangul": 1.3339, "cyrillic": 0.5288, "emoji": 2.9113, "other": 0.3672}
	gpt41 := TokenRates{"han": 0.9369, "latin": 0.2272, "digit": 0.8867, "hangul": 0.7902, "cyrillic": 0.2831, "emoji": 2.2016, "other": 0.1561}
	llama := TokenRates{"han": 0.8905, "latin": 0.2439, "digit": 0.7827, "hangul": 0.7994, "cyrillic": 0.3178, "emoji": 2.9855, "other": 0.1863}
	mistral := TokenRates{"han": 1.1036, "latin": 0.2798, "digit": 1.2948, "hangul": 1.4589, "cyrillic": 0.3861, "emoji": 3.1633, "other": 0.3988}
	qwen := TokenRates{"han": 0.6907, "latin": 0.2588, "digit": 1.177, "hangul": 0.8754, "cyrillic": 0.3571, "emoji": 1.3806, "other": 0.2598}

	add := func(name string, rates TokenRates) {
		BUILTIN_TOKEN_RATES[name] = rates
	}

	add("deepseek-chat", deepseek)
	add("deepseek-reasoner", deepseek)
	add("deepseek-v4-flash", deepseek)
	add("deepseek-v4-pro", deepseek)
	add("gpt-3.5-turbo", gpt3)
	add("gpt-4", gpt3)
	add("gpt-4-turbo", gpt3)
	add("gpt-4.1", gpt41)
	add("gpt-4.1-mini", gpt41)
	add("gpt-4o", gpt41)
	add("gpt-4o-mini", gpt41)
	add("gpt-5-mini", gpt41)
	add("gpt-5.4", gpt41)
	add("gpt-5.4-mini", gpt41)
	add("gpt-5.5", gpt41)
	add("llama-3.1-70b", llama)
	add("llama-3.1-8b", llama)
	add("mistral-7b", mistral)
	add("mistral-small", mistral)
	add("o1", gpt41)
	add("o3", gpt41)
	add("qwen-max", qwen)
	add("qwen-plus", qwen)
	add("qwen2.5-72b", qwen)
	add("qwen2.5-7b", qwen)
}
