import { TokenRates } from './buckets.js';

export const BUILTIN_TOKEN_RATES: Record<string, TokenRates> = {
  "deepseek-chat": {
    "han": 0.6896,
    "latin": 0.2521,
    "digit": 0.838,
    "hangul": 0.8542,
    "cyrillic": 0.2955,
    "emoji": 2.0485,
    "other": 0.1799
  },
  "deepseek-reasoner": {
    "han": 0.6896,
    "latin": 0.2521,
    "digit": 0.838,
    "hangul": 0.8542,
    "cyrillic": 0.2955,
    "emoji": 2.0485,
    "other": 0.1799
  },
  "deepseek-v4-flash": {
    "han": 0.6896,
    "latin": 0.2521,
    "digit": 0.838,
    "hangul": 0.8542,
    "cyrillic": 0.2955,
    "emoji": 2.0485,
    "other": 0.1799
  },
  "deepseek-v4-pro": {
    "han": 0.6896,
    "latin": 0.2521,
    "digit": 0.838,
    "hangul": 0.8542,
    "cyrillic": 0.2955,
    "emoji": 2.0485,
    "other": 0.1799
  },
  "gpt-3.5-turbo": {
    "han": 1.2812,
    "latin": 0.2458,
    "digit": 0.7706,
    "hangul": 1.3339,
    "cyrillic": 0.5288,
    "emoji": 2.9113,
    "other": 0.3672
  },
  "gpt-4": {
    "han": 1.2812,
    "latin": 0.2458,
    "digit": 0.7706,
    "hangul": 1.3339,
    "cyrillic": 0.5288,
    "emoji": 2.9113,
    "other": 0.3672
  },
  "gpt-4-turbo": {
    "han": 1.2812,
    "latin": 0.2458,
    "digit": 0.7706,
    "hangul": 1.3339,
    "cyrillic": 0.5288,
    "emoji": 2.9113,
    "other": 0.3672
  },
  "gpt-4.1": {
    "han": 0.9369,
    "latin": 0.2272,
    "digit": 0.8867,
    "hangul": 0.7902,
    "cyrillic": 0.2831,
    "emoji": 2.2016,
    "other": 0.1561
  },
  "gpt-4.1-mini": {
    "han": 0.9369,
    "latin": 0.2272,
    "digit": 0.8867,
    "hangul": 0.7902,
    "cyrillic": 0.2831,
    "emoji": 2.2016,
    "other": 0.1561
  },
  "gpt-4o": {
    "han": 0.9369,
    "latin": 0.2272,
    "digit": 0.8867,
    "hangul": 0.7902,
    "cyrillic": 0.2831,
    "emoji": 2.2016,
    "other": 0.1561
  },
  "gpt-4o-mini": {
    "han": 0.9369,
    "latin": 0.2272,
    "digit": 0.8867,
    "hangul": 0.7902,
    "cyrillic": 0.2831,
    "emoji": 2.2016,
    "other": 0.1561
  },
  "gpt-5-mini": {
    "han": 0.9369,
    "latin": 0.2272,
    "digit": 0.8867,
    "hangul": 0.7902,
    "cyrillic": 0.2831,
    "emoji": 2.2016,
    "other": 0.1561
  },
  "gpt-5.4": {
    "han": 0.9369,
    "latin": 0.2272,
    "digit": 0.8867,
    "hangul": 0.7902,
    "cyrillic": 0.2831,
    "emoji": 2.2016,
    "other": 0.1561
  },
  "gpt-5.4-mini": {
    "han": 0.9369,
    "latin": 0.2272,
    "digit": 0.8867,
    "hangul": 0.7902,
    "cyrillic": 0.2831,
    "emoji": 2.2016,
    "other": 0.1561
  },
  "gpt-5.5": {
    "han": 0.9369,
    "latin": 0.2272,
    "digit": 0.8867,
    "hangul": 0.7902,
    "cyrillic": 0.2831,
    "emoji": 2.2016,
    "other": 0.1561
  },
  "llama-3.1-70b": {
    "han": 0.8905,
    "latin": 0.2439,
    "digit": 0.7827,
    "hangul": 0.7994,
    "cyrillic": 0.3178,
    "emoji": 2.9855,
    "other": 0.1863
  },
  "llama-3.1-8b": {
    "han": 0.8905,
    "latin": 0.2439,
    "digit": 0.7827,
    "hangul": 0.7994,
    "cyrillic": 0.3178,
    "emoji": 2.9855,
    "other": 0.1863
  },
  "mistral-7b": {
    "han": 1.1036,
    "latin": 0.2798,
    "digit": 1.2948,
    "hangul": 1.4589,
    "cyrillic": 0.3861,
    "emoji": 3.1633,
    "other": 0.3988
  },
  "mistral-small": {
    "han": 1.1036,
    "latin": 0.2798,
    "digit": 1.2948,
    "hangul": 1.4589,
    "cyrillic": 0.3861,
    "emoji": 3.1633,
    "other": 0.3988
  },
  "o1": {
    "han": 0.9369,
    "latin": 0.2272,
    "digit": 0.8867,
    "hangul": 0.7902,
    "cyrillic": 0.2831,
    "emoji": 2.2016,
    "other": 0.1561
  },
  "o3": {
    "han": 0.9369,
    "latin": 0.2272,
    "digit": 0.8867,
    "hangul": 0.7902,
    "cyrillic": 0.2831,
    "emoji": 2.2016,
    "other": 0.1561
  },
  "qwen-max": {
    "han": 0.6907,
    "latin": 0.2588,
    "digit": 1.177,
    "hangul": 0.8754,
    "cyrillic": 0.3571,
    "emoji": 1.3806,
    "other": 0.2598
  },
  "qwen-plus": {
    "han": 0.6907,
    "latin": 0.2588,
    "digit": 1.177,
    "hangul": 0.8754,
    "cyrillic": 0.3571,
    "emoji": 1.3806,
    "other": 0.2598
  },
  "qwen2.5-72b": {
    "han": 0.6907,
    "latin": 0.2588,
    "digit": 1.177,
    "hangul": 0.8754,
    "cyrillic": 0.3571,
    "emoji": 1.3806,
    "other": 0.2598
  },
  "qwen2.5-7b": {
    "han": 0.6907,
    "latin": 0.2588,
    "digit": 1.177,
    "hangul": 0.8754,
    "cyrillic": 0.3571,
    "emoji": 1.3806,
    "other": 0.2598
  }
};