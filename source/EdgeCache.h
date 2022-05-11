#pragma once

#include "MPIWrapper.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

template<typename EdgeType>
class EdgeCache {

	std::vector<std::map<int, std::vector<EdgeType>>> cache{};

public:
	void init() {
		const auto number_ranks = MPIWrapper::get_number_ranks();
		cache.resize(number_ranks);
	}

	void clear() {
		for (auto& entry : cache) {
			entry.clear();
		}
	}

	bool contains(int rank, int node_id) {
		const auto& entry = cache[rank];
		const auto& where_it = entry.find(node_id);

		return where_it != entry.cend();
	}

	std::vector<EdgeType> get_value(int rank, int node_id) {
		return cache[rank][node_id];
	}

	void insert(int rank, int node_id, std::vector<EdgeType> value) {
		auto& entry = cache[rank];
		entry.emplace(node_id, std::move(value));
	}
};
