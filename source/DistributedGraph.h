#pragma once

#include "MPIWrapper.h"
#include "Edge.h"
#include "EdgeCache.h"
#include "Vec3.h"

#include <cstdint>
#include <filesystem>
#include <vector>

class DistributedGraph {
public:
	DistributedGraph(const std::filesystem::path& path);

	Vec3d get_node_position(int mpi_rank, int node_id) const;

	std::uint64_t get_number_in_edges(int mpi_rank, int node_id) const;

	std::uint64_t get_number_out_edges(int mpi_rank, int node_id) const;

	InEdge get_in_edge(int mpi_rank, int node_id, int edge_id) const;

	std::vector<InEdge> get_in_edges(int mpi_rank, int node_id) const;

	OutEdge get_out_edge(int mpi_rank, int node_id, int edge_id) const;

	std::vector<OutEdge> get_out_edges(int mpi_rank, int node_id) const;

	std::uint64_t get_number_local_nodes() const noexcept {
		return local_number_nodes;
	}

	std::uint64_t get_number_local_in_edges() const noexcept {
		return local_number_in_edges;
	}

	std::uint64_t get_number_local_out_edges() const noexcept {
		return local_number_out_edges;
	}

	void lock_all_rma_windows() const;

	void unlock_all_rma_windows() const;

private:
	void load_nodes(const std::filesystem::path& path);

	void load_in_edges(const std::filesystem::path& path);

	void load_out_edges(const std::filesystem::path& path);

	RMAWindow<Vec3d> nodes_window{};

	RMAWindow<InEdge> in_edges_window{};
	RMAWindow<std::uint64_t> prefix_in_edges_window{};
	RMAWindow<std::uint64_t> number_in_edges_window{};

	RMAWindow<OutEdge> out_edges_window{};
	RMAWindow<std::uint64_t> prefix_out_edges_window{};
	RMAWindow<std::uint64_t> number_out_edges_window{};

	mutable EdgeCache<InEdge> in_edge_cache{};
	mutable EdgeCache<OutEdge> out_edge_cache{};

	std::uint64_t local_number_nodes{};
	std::uint64_t local_number_in_edges{};
	std::uint64_t local_number_out_edges{};
};