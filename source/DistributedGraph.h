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

		Vec3d position;
		MPIWrapper::passive_rget(&position,sizeof(Vec3d),MPI_BYTE,mpi_rank,node_id*sizeof(Vec3d),
								 sizeof(Vec3d),MPI_BYTE, nodes_window.window);

		return position;
	}

	std::uint64_t get_number_in_edges(int mpi_rank, int node_id) const {
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank) {
			const auto* const number_ptr = number_in_edges_window.my_base_pointer;
			const auto number = number_ptr[node_id];
			return number;
		}

		std::uint64_t number_in_edges;
		MPIWrapper::passive_rget(&number_in_edges,1,MPI_UINT64_T,mpi_rank,node_id*sizeof(MPI_UINT64_T),
								 1,MPI_UINT64_T, number_in_edges_window.window);
		return number_in_edges;
	}

	std::uint64_t get_number_out_edges(int mpi_rank, int node_id) const {
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank) {
			const auto* const number_ptr = number_out_edges_window.my_base_pointer;
			const auto number = number_ptr[node_id];
			return number;
		}
		
		std::cout<<"Nonlocal get_number_out_edges "<<mpi_rank<<" "<<node_id<<std::endl;

		std::uint64_t number_out_edges;
		MPIWrapper::passive_rget(&number_out_edges,1,MPI_UINT64_T,mpi_rank,node_id*sizeof(MPI_UINT64_T),
								 1,MPI_UINT64_T, number_out_edges_window.window);

		std::cout<<" -- Nonlocal get_number_out_edges "<<mpi_rank<<" "<<node_id<<std::endl;

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

		std::uint64_t prefix_in_edge;
		MPIWrapper::passive_rget(&prefix_in_edge,1,MPI_UINT64_T,mpi_rank,node_id*sizeof(MPI_UINT64_T),
								 1,MPI_UINT64_T, prefix_in_edges_window.window);
			
		InEdge edge;
		MPIWrapper::passive_rget(&edge,sizeof(InEdge),MPI_BYTE,mpi_rank,(prefix_in_edge+edge_id)*sizeof(InEdge),
								 1,MPI_BYTE,in_edges_window.window);

		return edge;
	}

	const std::vector<InEdge>& get_in_edges(int mpi_rank, int node_id) const
	{
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank)
		{
			const auto* const prefix_ptr = prefix_in_edges_window.my_base_pointer;
			const std::uint64_t prefix = prefix_ptr[node_id];

			const std::uint64_t number_in_edges = get_number_in_edges(mpi_rank,node_id);
			std::vector<InEdge> edges(number_in_edges);

			const auto* const edge_ptr = in_edges_window.my_base_pointer;
			for(int edge_id=0;edge_id<number_in_edges;edge_id++)
			{
				edges[edge_id] = edge_ptr[prefix + edge_id];
			}
			in_edge_cache.insert(mpi_rank, node_id, std::move(edges));
		}
		else if (!in_edge_cache.contains(mpi_rank, node_id))
		{
			std::cout<<"Nonlocal in edge"<<std::endl;
			
			std::uint64_t prefix_in_edge;
			MPIWrapper::passive_rget(&prefix_in_edge,1,MPI_UINT64_T,mpi_rank,node_id*sizeof(MPI_UINT64_T),
								 1,MPI_UINT64_T, prefix_in_edges_window.window);
			
			const std::uint64_t number_in_edges = get_number_in_edges(mpi_rank, node_id);
			std::vector<InEdge> edges(number_in_edges);
			MPIWrapper::passive_rget(edges.data(),number_in_edges*sizeof(InEdge),MPI_BYTE,mpi_rank,
									 prefix_in_edge*sizeof(InEdge),number_in_edges*sizeof(InEdge),
									 MPI_BYTE,in_edges_window.window);
			
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

		std::uint64_t prefix_out_edge;
		MPIWrapper::passive_rget(&prefix_out_edge,1,MPI_UINT64_T,mpi_rank,node_id*sizeof(MPI_UINT64_T),
								 1,MPI_UINT64_T, prefix_out_edges_window.window);
			
		OutEdge edge;
		MPIWrapper::passive_rget(&edge,sizeof(OutEdge),MPI_BYTE,mpi_rank,(prefix_out_edge+edge_id)*sizeof(OutEdge),
								 1,MPI_BYTE,out_edges_window.window);

		return edge;
	}

	const std::vector<OutEdge>& get_out_edges(int mpi_rank, int node_id) const
	{
		if (const auto my_rank = MPIWrapper::get_my_rank(); my_rank == mpi_rank)
		{
			const auto* const prefix_ptr = prefix_out_edges_window.my_base_pointer;
			const std::uint64_t prefix = prefix_ptr[node_id];

			const std::uint64_t number_out_edges = get_number_out_edges(mpi_rank,node_id);
			std::vector<OutEdge> edges(number_out_edges);

			const auto* const edge_ptr = out_edges_window.my_base_pointer;
			for(int edge_id=0;edge_id<number_out_edges;edge_id++)
			{
				edges[edge_id] = edge_ptr[prefix + edge_id];
			}
			out_edge_cache.insert(mpi_rank, node_id, std::move(edges));
		}
		if (!out_edge_cache.contains(mpi_rank, node_id))
		{
			std::cout<<"Nonlocal out edge "<<std::endl;
			
			std::uint64_t prefix_out_edge;
			MPIWrapper::passive_rget(&prefix_out_edge,1,MPI_UINT64_T,mpi_rank,node_id*sizeof(MPI_UINT64_T),
								 1,MPI_UINT64_T, prefix_out_edges_window.window);
			
			const std::uint64_t number_out_edges = get_number_out_edges(mpi_rank, node_id);
			std::vector<OutEdge> edges(number_out_edges);
			MPIWrapper::passive_rget(edges.data(),number_out_edges*sizeof(OutEdge),MPI_BYTE,mpi_rank,
									 prefix_out_edge*sizeof(OutEdge),number_out_edges*sizeof(OutEdge),
									 MPI_BYTE,out_edges_window.window);
			
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
