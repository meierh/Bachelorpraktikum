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
		Vec3d position;
		MPIWrapper::passive_sync_RMA_get<Vec3d>(&position,1,node_id,mpi_rank,MPIWrapper::MPI_Vec3d,
												nodes_window);
		return position;
	}

	std::uint64_t get_node_area_localID(int mpi_rank, int node_id) const {
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank) {
			const auto* const area_names_ind_ptr = area_names_ind_window.my_base_pointer;
			const auto area_name_ind = area_names_ind_ptr[node_id];
			return area_name_ind;
		}

		std::uint64_t area_name_ind{};

		MPI_Request request_item{};

		const auto error_code = MPI_Rget(&area_name_ind, sizeof(std::uint64_t), MPI_BYTE, mpi_rank, sizeof(std::uint64_t) * node_id, sizeof(std::uint64_t), MPI_BYTE, area_names_ind_window.window, &request_item);
		MPI_Wait(&request_item, MPI_STATUS_IGNORE);

		if (error_code != MPI_SUCCESS) {
			std::cout << "Fetching a remote value returned the error code: " << error_code << std::endl;
			throw error_code;
		}

		return area_name_ind;
	}
	
	std::uint64_t get_node_signal_localID(int mpi_rank, int node_id) const {
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank) {
			const auto* const signal_types_ind_ptr = signal_types_ind_window.my_base_pointer;
			const auto signal_type_ind = signal_types_ind_ptr[node_id];
			return signal_type_ind;
		}

		std::uint64_t signal_type_ind{};

		MPI_Request request_item{};

		const auto error_code = MPI_Rget(&signal_type_ind, sizeof(std::uint64_t), MPI_BYTE, mpi_rank, sizeof(std::uint64_t) * node_id, sizeof(std::uint64_t), MPI_BYTE, signal_types_ind_window.window, &request_item);
		MPI_Wait(&request_item, MPI_STATUS_IGNORE);

		if (error_code != MPI_SUCCESS) {
			std::cout << "Fetching a remote value returned the error code: " << error_code << std::endl;
			throw error_code;
		}

		return signal_type_ind;
	}
	
	std::uint64_t get_number_in_edges(int mpi_rank, int node_id) const {
		std::uint64_t number_in_edges;
		MPIWrapper::passive_sync_RMA_get<std::uint64_t>(&number_in_edges,1,node_id,mpi_rank,MPI_UINT64_T,
														number_in_edges_window);
		return number_in_edges;
	}

	std::uint64_t get_number_out_edges(int mpi_rank, int node_id) const {
		std::uint64_t number_out_edges;
		MPIWrapper::passive_sync_RMA_get<std::uint64_t>(&number_out_edges,1,node_id,mpi_rank,MPI_UINT64_T,
														number_out_edges_window);
		return number_out_edges;
	}

	InEdge get_in_edge(int mpi_rank, int node_id, int edge_id) const {
		std::uint64_t prefix_in_edge;
		MPIWrapper::passive_sync_RMA_get<std::uint64_t>(&prefix_in_edge,1,node_id,mpi_rank,MPI_UINT64_T,
														prefix_in_edges_window);
		InEdge edge;
		MPIWrapper::passive_sync_RMA_get<InEdge>(&edge,1,prefix_in_edge+edge_id,mpi_rank,MPIWrapper::MPI_InEdge,
												 in_edges_window);
		return edge;
	}

	const std::vector<InEdge>& get_in_edges(int mpi_rank, int node_id) const
	{
		if(!in_edge_cache.contains(mpi_rank, node_id))
		{
			std::uint64_t prefix_in_edge;
			MPIWrapper::passive_sync_RMA_get<std::uint64_t>(&prefix_in_edge,1,node_id,mpi_rank,MPI_UINT64_T,
															prefix_in_edges_window);
				
			const std::uint64_t number_in_edges = get_number_in_edges(mpi_rank, node_id);
			std::vector<InEdge> edges(number_in_edges);
			MPIWrapper::passive_sync_RMA_get<InEdge>(edges.data(),number_in_edges,prefix_in_edge,mpi_rank,
													 MPIWrapper::MPI_InEdge,in_edges_window);
			in_edge_cache.insert(mpi_rank, node_id, std::move(edges));
		}
		return in_edge_cache.get_value(mpi_rank, node_id);
	}

	OutEdge get_out_edge(int mpi_rank, int node_id, int edge_id) const
	{
		std::uint64_t prefix_out_edge;
		MPIWrapper::passive_sync_RMA_get<std::uint64_t>(&prefix_out_edge,1,node_id,mpi_rank,MPI_UINT64_T,
														prefix_out_edges_window);
		
		OutEdge edge;
		MPIWrapper::passive_sync_RMA_get<OutEdge>(&edge,1,prefix_out_edge+edge_id,mpi_rank,MPIWrapper::MPI_OutEdge,
												 out_edges_window);
		return edge;
	}

	const std::vector<OutEdge>& get_out_edges(int mpi_rank, int node_id) const
	{
		if (!out_edge_cache.contains(mpi_rank, node_id))
		{
			std::uint64_t prefix_out_edge;
			MPIWrapper::passive_sync_RMA_get<std::uint64_t>(&prefix_out_edge,1,node_id,mpi_rank,MPI_UINT64_T,
															prefix_out_edges_window);
			const std::uint64_t number_out_edges = get_number_out_edges(mpi_rank, node_id);
			std::vector<OutEdge> edges(number_out_edges);
			MPIWrapper::passive_sync_RMA_get<OutEdge>(edges.data(),number_out_edges,prefix_out_edge,mpi_rank,
													  MPIWrapper::MPI_OutEdge,out_edges_window);
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

	const std::vector<std::string>& get_area_names() const noexcept {
		return area_names;
	}
	
	void lock_all_rma_windows() const;

	void unlock_all_rma_windows() const;

private:
	void load_nodes(const std::filesystem::path& path);

	void load_in_edges(const std::filesystem::path& path);

	void load_out_edges(const std::filesystem::path& path);

	RMAWindow<Vec3d> nodes_window{};
	RMAWindow<std::uint64_t> area_names_ind_window{};
	RMAWindow<std::uint64_t> signal_types_ind_window{};	
	std::vector<std::string> area_names;
	std::vector<std::string> signal_types;

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
