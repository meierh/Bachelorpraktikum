#pragma once

#include "mpi.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>
#include <stdexcept>

#include <sstream>
#include <iostream>

#include "Edge.h"
//#include "GraphProperty.h"

inline int do_nothing(void*) {
	return 0;
}

template<typename T = void>
struct RMAWindow {
	MPI_Win window{};
	std::uint64_t size{};
	std::vector<T*> base_pointers{};
	T* my_base_pointer{};

	RMAWindow() {}

	RMAWindow(T* ptr) : my_base_pointer(ptr) {

	}

	RMAWindow(const RMAWindow& other) = default;
	RMAWindow(RMAWindow&& other) = default;

	RMAWindow& operator=(const RMAWindow& other) = default;
	RMAWindow& operator=(RMAWindow&& other) = default;
};

class MPIWrapper {
	inline static int number_ranks{ -1 };
	inline static int my_rank{ -1 };

	inline static std::vector<MPI_Win> windows{};
	inline static std::vector<void*> memories{};

	inline static MPI_Op vector_sum_op{};

	static void vector_sum_function(void* in, void* in_out, int* len, MPI_Datatype* datatype) {
		auto* in_double = static_cast<double*>(in);
		auto* in_out_double = static_cast<double*>(in_out);

		int length = *len;

		for (auto i = 0; i < length; i++) {
			in_out_double[i] += in_double[i];
		}
	}
	
	static void create_mpi_datatypes(){
		//MPI Type for Vec3d 
		MPI_Type_contiguous(3,MPI_DOUBLE,&MPI_Vec3d);
		MPI_Type_commit(&MPI_Vec3d);
		
		//MPI Type for MPI_stdPair_of_AreaLocalID 
		MPI_Type_contiguous(4,MPI_UINT64_T,&MPI_stdPair_of_AreaLocalID);
		MPI_Type_commit(&MPI_stdPair_of_AreaLocalID);
		
		//MPI Type for InEdge
		InEdge iEdge;
		int         numberofValuesPerStructElement1[3] = {1,1,1};
		MPI_Aint    displacementOfStructElements1[3];
		MPI_Aint    base_address1;
		MPI_Get_address(&iEdge, &base_address1);
		MPI_Get_address(&iEdge.source_rank, &displacementOfStructElements1[0]);
		MPI_Get_address(&iEdge.source_id,   &displacementOfStructElements1[1]);
		MPI_Get_address(&iEdge.weight, 		&displacementOfStructElements1[2]);
		displacementOfStructElements1[0] = MPI_Aint_diff(displacementOfStructElements1[0], base_address1);
		displacementOfStructElements1[1] = MPI_Aint_diff(displacementOfStructElements1[1], base_address1);
		displacementOfStructElements1[2] = MPI_Aint_diff(displacementOfStructElements1[2], base_address1);
		MPI_Datatype typesOfStructElements1[3] = {MPI_INT,MPI_UNSIGNED,MPI_INT};
		MPI_Type_create_struct
		(
			3,
			numberofValuesPerStructElement1,
			displacementOfStructElements1,
			typesOfStructElements1,
			&MPI_InEdge
		);
		MPI_Type_commit(&MPI_InEdge);
		
		//MPI Type for OutEdge
		OutEdge oEdge;
		int         numberofValuesPerStructElement2[3] = {1,1,1};
		MPI_Aint    displacementOfStructElements2[3];
		MPI_Aint    base_address2;
		MPI_Get_address(&oEdge, &base_address2);
		MPI_Get_address(&oEdge.target_rank, &displacementOfStructElements2[0]);
		MPI_Get_address(&oEdge.target_id,   &displacementOfStructElements2[1]);
		MPI_Get_address(&oEdge.weight, 		&displacementOfStructElements2[2]);
		displacementOfStructElements2[0] = MPI_Aint_diff(displacementOfStructElements2[0], base_address2);
		displacementOfStructElements2[1] = MPI_Aint_diff(displacementOfStructElements2[1], base_address2);
		displacementOfStructElements2[2] = MPI_Aint_diff(displacementOfStructElements2[2], base_address2);
		MPI_Datatype typesOfStructElements2[3] = {MPI_INT,MPI_UNSIGNED,MPI_INT};
		MPI_Type_create_struct
		(
			3,
			numberofValuesPerStructElement2,
			displacementOfStructElements2,
			typesOfStructElements2,
			&MPI_OutEdge
		);
		MPI_Type_commit(&MPI_OutEdge);
		
		//MPI Type for AreaConnectivityInfo
		MPI_Type_contiguous(5,MPI_INT64_T,&MPI_AreaConnectivityInfo);
		MPI_Type_commit(&MPI_AreaConnectivityInfo);
		
		//MPI Type for threeMotifStructure
		MPI_Type_contiguous(7,MPI_UINT64_T,&MPI_threeMotifStructure);
		MPI_Type_commit(&MPI_threeMotifStructure);
		
		//MPI Type for nodeModularityInfo 
		MPI_Type_contiguous(3,MPI_UINT64_T,&MPI_nodeModularityInfo);
		MPI_Type_commit(&MPI_nodeModularityInfo);
		
		//MPI Type for EdgeDegrees 
		MPI_Type_contiguous(4,MPI_UINT64_T,&MPI_EdgeDegrees);
		MPI_Type_commit(&MPI_EdgeDegrees);
	}

public:
	inline static MPI_Datatype MPI_Vec3d;
	inline static MPI_Datatype MPI_InEdge;
	inline static MPI_Datatype MPI_OutEdge;
	inline static MPI_Datatype MPI_stdPair_of_AreaLocalID;
	inline static MPI_Datatype MPI_AreaConnectivityInfo;
	inline static MPI_Datatype MPI_threeMotifStructure;
	inline static MPI_Datatype MPI_nodeModularityInfo;
	inline static MPI_Datatype MPI_EdgeDegrees;
	
	static void init(int argument_count, char* arguments[]) {
		if (const auto error_code = MPI_Init(&argument_count, &arguments); error_code != 0) {
			std::cout << "Initializing MPI returned the error: " << error_code << std::endl;
			throw error_code;
		}

		if (const auto error_code = MPI_Comm_size(MPI_COMM_WORLD, &number_ranks); error_code != 0) {
			std::cout << "Fetching the communicator size returned the error: " << error_code << std::endl;
			throw error_code;
		}

		if (const auto error_code = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); error_code != 0) {
			std::cout << "Fetching my communicator id returned the error: " << error_code << std::endl;
			throw error_code;
		}

		if (const auto error_code = MPI_Op_create(vector_sum_function, 1, &vector_sum_op); error_code != 0) {
			std::cout << "Creating the user-defined operation returned the error: " << error_code << std::endl;
			throw error_code;
		}
		
		create_mpi_datatypes();
	}

	static void finalize() {
		barrier();

		for (auto window : windows) {
			MPI_Win_free(&window);
		}

		for (auto* ptr : memories) {
			MPI_Free_mem(ptr);
		}

		if (const auto error_code = MPI_Op_free(&vector_sum_op); error_code != 0) {
			std::cout << "Freeing the user-defined function returned the error: " << error_code;
			throw error_code;
		}

		if (const auto error_code = MPI_Finalize(); error_code != 0) {
			std::cout << "Finalizing MPI returned the error: " << error_code << std::endl;
			return;
		}
	}

	static int get_my_rank() noexcept {
		return my_rank;
	}

	static int get_number_ranks() noexcept {
		return number_ranks;
	}

	static void lock_window_exclusive(const int mpi_rank, MPI_Win window) {//MPI_MODE_NOCHECK
		if (const auto error_code = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, mpi_rank, 0, window); error_code != MPI_SUCCESS) {
			std::cout << "Exclusive-locking the RMA window returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}

	static void lock_window_shared(const int mpi_rank, MPI_Win window) { //MPI_MODE_NOCHECK
		if (const auto error_code = MPI_Win_lock(MPI_LOCK_SHARED, mpi_rank, 0, window); error_code != MPI_SUCCESS) {
			std::cout << "Shared-locking the RMA window returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}

	static void unlock_window(const int mpi_rank, MPI_Win window) {
		if (const auto error_code = MPI_Win_unlock(mpi_rank, window); error_code != MPI_SUCCESS) {
			std::cout << "Unlocking the RMA window returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}

	static void barrier() {
		if (const auto error_code = MPI_Barrier(MPI_COMM_WORLD); error_code != 0) {
			std::cout << "Synchronizing all MPI ranks returned the error: " << error_code << std::endl;
			return;
		}
	}

	template<typename T>
	static RMAWindow<T> create_rma_window(std::uint64_t number_elements) {
		const auto window_size = number_elements * sizeof(T);
		T* ptr = nullptr;

		if (const auto error_code = MPI_Alloc_mem(window_size, MPI_INFO_NULL, &ptr); error_code != 0) {
			std::cout << "Allocating the shared memory returned the error: " << error_code << std::endl;
			throw error_code;
		}

		RMAWindow<T> window(ptr);
		window.size = window_size;
		window.base_pointers.resize(number_ranks);

		if (const auto error_code = MPI_Win_create(ptr, window_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &window.window); error_code != 0) {
			std::cout << "Allocating the shared window returned the error: " << error_code << std::endl;
			throw error_code;
		}

		if (const auto error_code = MPI_Allgather(&ptr, 1, MPI_AINT, window.base_pointers.data(), 1, MPI_AINT, MPI_COMM_WORLD); error_code != 0) {
			std::cout << "Gathering all shared window pointers returned the error: " << error_code << std::endl;
			throw error_code;
		}

		windows.emplace_back(window.window);
		memories.emplace_back(ptr);

		return window;
	}

	static std::vector<std::uint64_t> all_gather(std::uint64_t own_data) {
		std::vector<std::uint64_t> results(number_ranks);

		if (const int error_code = MPI_Allgather(&own_data, 1, MPI_UINT64_T, results.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD); error_code != 0) {
			std::cout << "Gathering all values returned the error : " << error_code << std::endl;
			throw error_code;
		}

		return results;
	}

	static std::uint64_t reduce_sum(std::uint64_t value) {
		std::uint64_t total_value = 0;

		if (const auto error_code = MPI_Reduce(&value, &total_value, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD); error_code != 0) {
			std::cout << "Reducing all values returned the error: " << error_code << std::endl;
			throw error_code;
		}

		return total_value;
	}

	static std::vector<double> reduce_componentwise(const std::vector<double>& values) {
		const auto number_values = values.size();
		const auto minimum_number_values = all_reduce_min(number_values);

		if (number_values != minimum_number_values) {
			std::cout << "Reducing componentwise with differently sized vectors!\n";
			throw number_values;
		}

		std::vector<double> buffer(number_values, 0.0);
		if (const auto error_code = MPI_Reduce(values.data(), buffer.data(), number_values, MPI_DOUBLE, vector_sum_op, 0, MPI_COMM_WORLD); error_code != 0) {
			std::cout << "Reducing componentwise returned the error: " << error_code << std::endl;
			throw error_code;
		}

		return buffer;
	}

	static double reduce_sum(double value) {
		double total_value = 0;

		if (const auto error_code = MPI_Reduce(&value, &total_value, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); error_code != 0) {
			std::cout << "Reducing all values returned the error: " << error_code << std::endl;
			throw error_code;
		}

		return total_value;
	}

	static std::uint64_t all_reduce_sum(std::uint64_t value) {
		std::uint64_t total_value = 0;

		if (const auto error_code = MPI_Allreduce(&value, &total_value, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD); error_code != 0) {
			std::cout << "All-reducing all values returned the error: " << error_code << std::endl;
			throw error_code;
		}

		return total_value;
	}

	static std::uint64_t all_reduce_min(std::uint64_t value) {
		std::uint64_t total_value = 0;

		if (const auto error_code = MPI_Allreduce(&value, &total_value, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD); error_code != 0) {
			std::cout << "All-reducing all values returned the error: " << error_code << std::endl;
			throw error_code;
		}

		return total_value;
	}

	static double all_reduce_sum(double value) {
		double total_value = 0;

		if (const auto error_code = MPI_Allreduce(&value, &total_value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); error_code != 0) {
			std::cout << "All-reducing all values returned the error: " << error_code << std::endl;
			throw error_code;
		}

		return total_value;
	}

	static std::uint64_t reduce_min(std::uint64_t value) {
		std::uint64_t total_value = 0;

		if (const auto error_code = MPI_Reduce(&value, &total_value, 1, MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD); error_code != 0) {
			std::cout << "Reducing all values returned the error: " << error_code << std::endl;
			throw error_code;
		}

		return total_value;
	}

	static std::uint64_t reduce_max(std::uint64_t value) {
		std::uint64_t total_value = 0;

		if (const auto error_code = MPI_Reduce(&value, &total_value, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD); error_code != 0) {
			std::cout << "Reducing all values returned the error: " << error_code << std::endl;
			throw error_code;
		}

		return total_value;
	}
	
	template<typename T>
	static void all_reduce(T* src, T* dest,int count,MPI_Datatype datatype,MPI_Op op) {
		const int error_code = MPI_Allreduce(src,dest,count,datatype,op,MPI_COMM_WORLD);
		if ( error_code != 0) {
			std::cout << "All-reducing all values returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}
	
	template<typename T>
	static void reduce(T* src, T* dest,int count,MPI_Datatype datatype,MPI_Op op,int root) {
		const int error_code = MPI_Reduce(src,dest,count,datatype,op,root,MPI_COMM_WORLD);
		if ( error_code != 0) {
			std::cout << "All-reducing all values returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}
	
	template<typename T>
	static void gather(T* src, T* dest, int count, MPI_Datatype datatype,int root){
		const int error_code= MPI_Gather(src,count,datatype,dest,count,datatype,root,MPI_COMM_WORLD);
		if (error_code != 0) {
			std::cout << "Gathering all values returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}
	
	template<typename T>
	static void all_gather(T* src, T* dest, int count, MPI_Datatype datatype){
		const int error_code= MPI_Allgather(src,count,datatype,dest,count,datatype,MPI_COMM_WORLD);
		if (error_code != 0) {
			std::cout << "All_Gathering all values returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}
	
	template<typename T>
	static void gatherv(T* src, int count, T* dest, int* destCounts, int* displs, MPI_Datatype datatype,int root){
		const int error_code= MPI_Gatherv(src,count,datatype,dest,destCounts,displs,datatype,root,MPI_COMM_WORLD);
		if (error_code != 0) {
			std::cout << "Gatherving all values returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}
	
	template<typename T>
	static void all_gatherv(T* src, int count, T* dest, int* destCounts, int* displs, MPI_Datatype datatype){
		const int error_code= MPI_Allgatherv(src,count,datatype,dest,destCounts,displs,datatype,MPI_COMM_WORLD);
		if (error_code != 0) {
			std::cout << "All_Gatherving all values returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}
	
	/* Access method for arbitrary rma window
	 * 
	 * Parameters: 
	 * 	dest_addr: Pointer to destination memory
	 * 	count: Number of transfered elements of type T/datatype
	 * 	src_disp: Displacement of initial transfer value
	 * 	src_rank: Rank of data to get
	 * 	datatype: MPI_Type corresponding to T
	 * 
	 * 	Requirements:
	 * 		[dest_addr,dest_addr+count*sizeof(T)) must be a valid memory range to write to
	 * 		[rma_window.my_base_pointer+src_disp*sizeof(T),rma_window.my_base_pointer+src_disp*sizeof(T)+count*sizeof(T)) 
	 * 			must be a valid memory range to read from to
	 * 		count >= 0
	 * 		src_disp >= 0
	 * 		src_rank in [0,number_of_ranks)
	 */
	template<typename T>
	static void passive_sync_rma_get(T *dest_addr, int count, int src_disp, int src_rank, MPI_Datatype datatype, const RMAWindow<T>& rma_window) {
		lock_window_shared(src_rank,rma_window.window);
		if(src_rank == my_rank) {	
			const T* const src_base_ptr = rma_window.my_base_pointer;
			const T* src_ptr = src_base_ptr+src_disp;
			std::memcpy(dest_addr,src_ptr,count*sizeof(T));
		}
		else {
			MPI_Request request_item;
			const int error_code = MPI_Rget(dest_addr, count, datatype, src_rank, src_disp*sizeof(T),
											count, datatype, rma_window.window, &request_item);
			if(error_code!=MPI_SUCCESS) {
					std::cout << "Fetching a remote value returned the error code: " << error_code << std::endl;
					throw error_code;
			}
			MPI_Wait(&request_item, MPI_STATUS_IGNORE);
		}
		unlock_window(src_rank,rma_window.window);
	}
};
