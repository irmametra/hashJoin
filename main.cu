#include "global.h"
#include "table.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cuda_profiler_api.h>

// GPU HashTable, atomicCAS
__global__ void build_HashTable_kernel(TupleR *dev_pr, TupleH *dev_ph) {
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < ST_SIZE) {
		uint32_t hk = LSB(dev_pr[x].key);
		while (atomicCAS((unsigned long long *)(dev_ph + hk), 0, *(unsigned long long *)(dev_pr + x)))
		{
					hk = (hk + 1) % H_SIZE;
		}
	}
}

// GPU histogram
__global__ void build_histogram_kernel(TupleS *dev_ps, TupleR *dev_ph,
		uint32_t *dev_histo) {
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < S_SIZE) {
		uint32_t hk = LSB(dev_ps[x].key);

		while (dev_ph[hk].key != 0) {
			if (dev_ph[hk].key == dev_ps[x].key) {
				++dev_histo[x];
			}
			hk = (hk + 1) % H_SIZE;
		}
	}
}

// GPU probe HashTable
__global__ void probe_HashTable_kernel(TupleS *dev_ps, TupleH *dev_ph,
		TupleT *dev_pt, uint32_t *dev_histo) {
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < S_SIZE) {
		TupleT *begin = dev_pt + dev_histo[x];
		TupleT *end = dev_pt + dev_histo[x + 1];
		uint32_t hk = LSB(dev_ps[x].key);
		while (begin != end) {
			if (dev_ph[hk].key == dev_ps[x].key) {
				*begin = TupleT(dev_ph[hk].r_id, dev_ps[x].s_id);
				++begin;
			}
			hk = (hk + 1) % H_SIZE;
		}
	}
}

int main() {
	CUDACheckError(cudaSetDevice(0));

	////////////////////////

	fprintf(stdout, "Number of 8-Byte Tuple Joined: %uMB\n", INPUT_SIZE);

	// TableR,id key
	TableR r;
	r.init();
	r.print();

	//TableS, MATCH_RATE id key
	TableS s;
	s.init(&r);
	s.print();

	// TableT, T_MAXSIZE
	TableT t;
	t.init();

	/////////////////////////

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	TupleR *dev_pr = 0;      // TableR in global memory
	TupleS *dev_ps = 0;      // TableS in global memory
	TupleT *dev_pt = 0;      // TalbeT in global memory
	TupleR *dev_ph = 0;      // HashTable in global memory
	uint32_t *dev_histo = 0; // Histogram in global memory

	CUDACheckError(cudaMalloc((void**) &dev_pr, R_SIZE * sizeof(TupleR)));
	CUDACheckError(cudaMalloc((void**) &dev_ps, S_SIZE * sizeof(TupleS)));
	CUDACheckError(cudaMalloc((void**) &dev_pt, T_MAXSIZE * sizeof(TupleT)));
	CUDACheckError(cudaMalloc((void**) &dev_ph, H_SIZE * sizeof(TupleH)));
	CUDACheckError(
			cudaMalloc((void**) &dev_histo, (S_SIZE + 1) * sizeof(uint32_t)));

	uint32_t ThreadPerBlock = 0;
	uint32_t BlockPerGrid = 0;

	cudaStream_t stream[NS];
	for (int i = 0; i < NS; i++) {
		cudaStreamCreate(&stream[i]);
	}

	cudaEventRecord(start, 0);

	//pinned memory
	CUDACheckError(
			cudaHostRegister(r.pr, R_SIZE * sizeof(TupleR), cudaHostAllocPortable));
	CUDACheckError(
			cudaHostRegister(s.ps, S_SIZE * sizeof(TupleS), cudaHostAllocPortable));
	CUDACheckError(
			cudaHostRegister(t.pt, T_MAXSIZE * sizeof(TupleT), cudaHostAllocPortable));

	//clear HashTable in global memory
	CUDACheckError(cudaMemset(dev_ph, 0, H_SIZE * sizeof(TupleH)));
	CUDACheckError(cudaMemset(dev_histo, 0, (S_SIZE + 1) * sizeof(uint32_t)));

	ThreadPerBlock = 1024;
	BlockPerGrid = (ST_SIZE - 1) / ThreadPerBlock + 1;

	// use streams overload memcpy and compute
	for (int i = 0; i < NS; i++) {
		CUDACheckError(
				cudaMemcpyAsync(dev_pr + i * ST_SIZE, r.pr + i * ST_SIZE, ST_SIZE * sizeof(TupleR), cudaMemcpyHostToDevice, stream[i]));
		CUDACheckError(
				cudaMemcpyAsync(dev_ps + i * ST_SIZE, s.ps + i * ST_SIZE, ST_SIZE * sizeof(TupleS), cudaMemcpyHostToDevice, stream[i]));
		build_HashTable_kernel<<<BlockPerGrid, ThreadPerBlock, 0, stream[i]>>>(
				dev_pr + i * ST_SIZE, dev_ph);
	}

	CUDACheckError(cudaGetLastError());
	CUDACheckError(cudaDeviceSynchronize());

	// build histogram in global memory
	ThreadPerBlock = 1024;
	BlockPerGrid = (S_SIZE - 1) / ThreadPerBlock + 1;
	build_histogram_kernel<<<BlockPerGrid, ThreadPerBlock>>>(dev_ps, dev_ph,
			dev_histo);
	CUDACheckError(cudaGetLastError());
	CUDACheckError(cudaDeviceSynchronize());

	// compute prefixsum
	// exclusive_scan
	thrust::device_ptr<uint32_t> dev_ptr = thrust::device_pointer_cast(
			dev_histo);
	thrust::exclusive_scan(dev_ptr, dev_ptr + S_SIZE + 1, dev_ptr);
	CUDACheckError(cudaGetLastError());
	CUDACheckError(cudaDeviceSynchronize());

	// probe HashTable
	ThreadPerBlock = 1024;
	BlockPerGrid = (S_SIZE - 1) / ThreadPerBlock + 1;
	probe_HashTable_kernel<<<BlockPerGrid, ThreadPerBlock>>>(dev_ps, dev_ph,
			dev_pt, dev_histo);
	CUDACheckError(cudaGetLastError());
	CUDACheckError(cudaDeviceSynchronize());

	// copy t_size
	CUDACheckError(
			cudaMemcpy(&(t.size), dev_histo + S_SIZE, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	// copy TableT to host memory
	CUDACheckError(
			cudaMemcpy(t.pt, dev_pt, t.size * sizeof(TupleT),
					cudaMemcpyDeviceToHost));

	CUDACheckError(cudaDeviceSynchronize());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	CUDACheckError(cudaFree(dev_pr));
	CUDACheckError(cudaFree(dev_ps));
	CUDACheckError(cudaFree(dev_pt));
	CUDACheckError(cudaFree(dev_ph));
	CUDACheckError(cudaFree(dev_histo));

	CUDACheckError(cudaGetLastError());
	CUDACheckError(cudaDeviceSynchronize());

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("GPU Max Clock rate: %.0f MHz (%0.2f GHz)\n",
			deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
	fprintf(stdout, "Your program executed time is %.1f ms.\n", elapsedTime);
	fprintf(stdout, "Your program executed %.0f GPU cycles.\n",
			elapsedTime * deviceProp.clockRate);
	fprintf(stdout, "Your program executed %.1f GPU cycles/tuple.\n",
			elapsedTime * deviceProp.clockRate / (R_SIZE + S_SIZE));

	CUDACheckError(cudaProfilerStop());

	t.print();
	t.check(&r, &s);

	r.clear();
	s.clear();
	t.clear();

	//Windows OS specific
	//system("pause");
	return 0;
}
