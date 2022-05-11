#pragma once

#include "mpi.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

inline int do_nothing(void*) {
	return 0;
}

template<typename T = void>
struct RMAWindow {
	MPI_Win window{};
	std::uint64_t size{};
	std::vector<T*> base_pointers{};
	T* my_base_pointer{};

	RMAWindow() { }

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

public:
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
	}

	static void finalize() {
		barrier();

		for (auto window : windows) {
			MPI_Win_free(&window);
		}

		for (auto* ptr : memories) {
			MPI_Free_mem(ptr);
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

	static void lock_window_exclusive(const int mpi_rank, MPI_Win window) {
		if (const auto error_code = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, mpi_rank, MPI_MODE_NOCHECK, window); error_code != MPI_SUCCESS) {
			std::cout << "Exclusive-locking the RMA window returned the error: " << error_code << std::endl;
			throw error_code;
		}
	}

	static void lock_window_shared(const int mpi_rank, MPI_Win window) {
		if (const auto error_code = MPI_Win_lock(MPI_LOCK_SHARED, mpi_rank, MPI_MODE_NOCHECK, window); error_code != MPI_SUCCESS) {
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
};
