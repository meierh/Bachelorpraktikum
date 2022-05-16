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

	Vec3d get_node_position(int mpi_rank, int node_id) const {
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank) {
			const auto* const pos_ptr = nodes_window.my_base_pointer;
			const auto position = pos_ptr[node_id];
			return position;
		}

		Vec3d position{};

		MPI_Request request_item{};

		const auto error_code = MPI_Rget(&position, sizeof(Vec3d), MPI_BYTE, mpi_rank, sizeof(Vec3d) * node_id, sizeof(Vec3d), MPI_BYTE, nodes_window.window, &request_item);
		MPI_Wait(&request_item, MPI_STATUS_IGNORE);

		if (error_code != MPI_SUCCESS) {
			std::cout << "Fetching a remote value returned the error code: " << error_code << std::endl;
			throw error_code;
		}

		return position;
	}

	std::uint64_t get_number_in_edges(int mpi_rank, int node_id) const {
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank) {
			const auto* const number_ptr = number_in_edges_window.my_base_pointer;
			const auto number = number_ptr[node_id];
			return number;
		}

		std::uint64_t number_in_edges{};

		MPI_Request request_item{};

		const auto error_code = MPI_Rget(&number_in_edges, 1, MPI_UINT64_T, mpi_rank, 8 * node_id, 1, MPI_UINT64_T, number_in_edges_window.window, &request_item);
		MPI_Wait(&request_item, MPI_STATUS_IGNORE);

		if (error_code != MPI_SUCCESS) {
			std::cout << "Fetching a remote value returned the error code: " << error_code << std::endl;
			throw error_code;
		}

		return number_in_edges;
	}

	std::uint64_t get_number_out_edges(int mpi_rank, int node_id) const {
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank) {
			const auto* const number_ptr = number_out_edges_window.my_base_pointer;
			const auto number = number_ptr[node_id];
			return number;
		}

		std::uint64_t number_out_edges{ 5555555 };

		MPI_Request request_item{};

		const auto error_code = MPI_Rget(&number_out_edges, 1, MPI_UINT64_T, mpi_rank, node_id * 8, 1, MPI_UINT64_T, number_out_edges_window.window, &request_item);
		MPI_Wait(&request_item, MPI_STATUS_IGNORE);

		if (error_code != MPI_SUCCESS) {
			std::cout << "Fetching a remote value returned the error code: " << error_code << std::endl;
			throw error_code;
		}

		return number_out_edges;
	}

	InEdge get_in_edge(int mpi_rank, int node_id, int edge_id) const {
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank) {
			const auto* const prefix_ptr = prefix_in_edges_window.my_base_pointer;
			const auto prefix = prefix_ptr[node_id];

			const auto* const edge_ptr = in_edges_window.my_base_pointer;
			const auto edge = edge_ptr[prefix + edge_id];
			return edge;
		}

		InEdge edge{};
		std::uint64_t prefix_in_edge{};

		MPI_Request request_item{};

		const auto error_code_1 = MPI_Rget(&prefix_in_edge, 1, MPI_UINT64_T, mpi_rank, 8 * node_id, 1, MPI_UINT64_T, prefix_in_edges_window.window, &request_item);
		MPI_Wait(&request_item, MPI_STATUS_IGNORE);

		if (error_code_1 != MPI_SUCCESS) {
			std::cout << "Fetching a remote value returned the error code: " << error_code_1 << std::endl;
			throw error_code_1;
		}

		const auto error_code_2 = MPI_Rget(&edge, sizeof(InEdge), MPI_BYTE, mpi_rank, prefix_in_edge * sizeof(InEdge), sizeof(InEdge), MPI_BYTE, in_edges_window.window, &request_item);
		MPI_Wait(&request_item, MPI_STATUS_IGNORE);

		if (error_code_2 != MPI_SUCCESS) {
			std::cout << "Fetching a remote value returned the error code: " << error_code_2 << std::endl;
			throw error_code_2;
		}

		return edge;
	}

	const std::vector<InEdge>& get_in_edges(int mpi_rank, int node_id) const {
		if (!in_edge_cache.contains(mpi_rank, node_id)) {
			const auto number_in_edges = get_number_in_edges(mpi_rank, node_id);

			std::vector<InEdge> edges{};
			edges.resize(number_in_edges);

			std::uint64_t prefix_in_edge{};

			MPI_Request request_item{};

			const auto error_code_1 = MPI_Rget(&prefix_in_edge, 1, MPI_UINT64_T, mpi_rank, 8 * node_id, 1, MPI_UINT64_T, prefix_in_edges_window.window, &request_item);
			MPI_Wait(&request_item, MPI_STATUS_IGNORE);

			if (error_code_1 != MPI_SUCCESS) {
				std::cout << "Fetching a remote value returned the error code: " << error_code_1 << std::endl;
				throw error_code_1;
			}

			const auto error_code_2 = MPI_Rget(edges.data(), sizeof(InEdge) * number_in_edges, MPI_BYTE, mpi_rank, prefix_in_edge * sizeof(InEdge), sizeof(InEdge) * number_in_edges, MPI_BYTE, in_edges_window.window, &request_item);
			MPI_Wait(&request_item, MPI_STATUS_IGNORE);

			if (error_code_2 != MPI_SUCCESS) {
				std::cout << "Fetching a remote value returned the error code: " << error_code_2 << std::endl;
				throw error_code_2;
			}

			in_edge_cache.insert(mpi_rank, node_id, std::move(edges));
		}

		return in_edge_cache.get_value(mpi_rank, node_id);
	}

	OutEdge get_out_edge(int mpi_rank, int node_id, int edge_id) const {
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank) {
			const auto* const prefix_ptr = prefix_out_edges_window.my_base_pointer;
			const auto prefix = prefix_ptr[node_id];

			const auto* const edge_ptr = out_edges_window.my_base_pointer;
			const auto edge = edge_ptr[prefix + edge_id];
			return edge;
		}

		OutEdge edge{};
		std::uint64_t prefix_out_edge{};

		MPI_Request request_item{};

		const auto error_code_1 = MPI_Rget(&prefix_out_edge, 1, MPI_UINT64_T, mpi_rank, 8 * node_id, 1, MPI_UINT64_T, prefix_out_edges_window.window, &request_item);
		MPI_Wait(&request_item, MPI_STATUS_IGNORE);

		if (error_code_1 != MPI_SUCCESS) {
			std::cout << "Fetching a remote value returned the error code: " << error_code_1 << std::endl;
			throw error_code_1;
		}

		const auto error_code_2 = MPI_Rget(&edge, sizeof(OutEdge), MPI_BYTE, mpi_rank, prefix_out_edge * sizeof(OutEdge), sizeof(OutEdge), MPI_BYTE, out_edges_window.window, &request_item);
		MPI_Wait(&request_item, MPI_STATUS_IGNORE);

		if (error_code_2 != MPI_SUCCESS) {
			std::cout << "Fetching a remote value returned the error code: " << error_code_2 << std::endl;
			throw error_code_2;
		}

		return edge;
	}

	const std::vector<OutEdge>& get_out_edges(int mpi_rank, int node_id) const {
		if (!out_edge_cache.contains(mpi_rank, node_id)) {
			const auto number_out_edges = get_number_out_edges(mpi_rank, node_id);

			std::vector<OutEdge> edges{};
			edges.resize(number_out_edges);

			std::uint64_t prefix_out_edge{};

			MPI_Request request_item{};

			const auto error_code_1 = MPI_Rget(&prefix_out_edge, 1, MPI_UINT64_T, mpi_rank, 8 * node_id, 1, MPI_UINT64_T, prefix_out_edges_window.window, &request_item);
			MPI_Wait(&request_item, MPI_STATUS_IGNORE);

			if (error_code_1 != MPI_SUCCESS) {
				std::cout << "Fetching a remote value returned the error code: " << error_code_1 << std::endl;
				throw error_code_1;
			}

			const auto error_code_2 = MPI_Rget(edges.data(), sizeof(InEdge) * number_out_edges, MPI_BYTE, mpi_rank, prefix_out_edge * sizeof(InEdge), sizeof(InEdge) * number_out_edges, MPI_BYTE, out_edges_window.window, &request_item);
			MPI_Wait(&request_item, MPI_STATUS_IGNORE);

			if (error_code_2 != MPI_SUCCESS) {
				std::cout << "Fetching a remote value returned the error code: " << error_code_2 << std::endl;
				throw error_code_2;
			}

			out_edge_cache.insert(mpi_rank, node_id, std::move(edges));
		}

		return out_edge_cache.get_value(mpi_rank, node_id);
	}

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