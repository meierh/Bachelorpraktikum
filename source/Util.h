#pragma once

#include <cstdint>
#include <vector>

inline std::vector<std::uint64_t> calculate_prefix_sum(const std::vector<std::uint64_t>& values) {
	std::vector<std::uint64_t> prefix_sum(values.size(), 0);

	for (auto i = 0; i < values.size() - 1; i++) {
		prefix_sum[i + 1] = prefix_sum[i] + values[i];
	}

	return prefix_sum;
}
