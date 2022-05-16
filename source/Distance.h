#pragma once

#include <cstdint>
#include <vector>
#include <utility>

struct VertexDistance {
	int mpi_rank;
	int node_id;
	unsigned int distance;
};

template <>
struct std::greater<VertexDistance> {
	bool operator()(const VertexDistance& lhs, const VertexDistance& rhs) const {
		return lhs.distance > rhs.distance;
	}
};

class NodePath {
	std::vector<std::uint64_t> nodes{};

public:
	NodePath(size_t reserved_capacity) {
		nodes.reserve(reserved_capacity);
	}

	void append_node(std::uint64_t node) {
		nodes.emplace_back(node);
	}

	const std::vector<std::uint64_t>& get_nodes() const noexcept {
		return nodes;
	}

};

struct VertexDistancePath {
	int mpi_rank;
	int node_id;
	unsigned int distance;
	NodePath path;
};

template <>
struct std::greater<VertexDistancePath> {
	bool operator()(const VertexDistancePath& lhs, const VertexDistancePath& rhs) const {
		return lhs.distance > rhs.distance;
	}
};

