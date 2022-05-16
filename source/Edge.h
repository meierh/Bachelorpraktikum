#pragma once

#include <cstdint>

struct InEdge {
	int source_rank{};
	unsigned int source_id{};
	int weight{};
	
	InEdge() = default;

	InEdge(int src_rank, unsigned int src_id, int wei) : source_rank(src_rank), source_id(src_id), weight(wei) {

	}
};

struct OutEdge {
	int target_rank{};
	unsigned int  target_id{};
	int weight{};

	OutEdge() = default;

	OutEdge(int tgt_rank, unsigned int tgt_id, int wei) : target_rank(tgt_rank), target_id(tgt_id), weight(wei) {

	}
};


