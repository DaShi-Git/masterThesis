#include <torch/extension.h>
#include <cuda.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <cuMat/src/DevicePointer.h>
#include <cuMat/src/Macros.h>
//#include "imodule.h"
#include "helper_math.cuh"
//#include "transfer_function.h"
#include "pytorch_utils.h"
#include "renderer_commons.cuh"
#include "kernel_loader.h"
#include "renderer_tensor.cuh"

#include <cuda_fp16.h>
#include <assert.h>

//#include <c10/util/Half.h>





namespace py = pybind11;


torch::Tensor evaluate_flexible_MLP(
	torch::Tensor& test, torch::Tensor& weights, const torch::Tensor& input, const torch::Tensor& bias, 
  const std::list <std::string>& hiddenStructure, const int batchsizeTotal, const int featuresize, const int outputdim0, const int outputdim2, const std::list <std::string>& activation, const int blockdim, const int griddim)
{
  CUstream stream = c10::cuda::getCurrentCUDAStream();
  
	auto scalarType = weights.scalar_type();
	
	const std::string kernelName = "EvaluateMLPFlexible";
	std::vector<std::string> constantNames;
	// if (const auto c = getConstantDeclarationName(s); !c.empty())
	// 	constantNames.push_back(c);
	std::stringstream extraSource;
  extraSource << "#include <cuda_fp16.h>"
		//<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
  for (auto const &i: activation) {
        extraSource << i << "\n";
    }
  for (auto const &i: hiddenStructure) {
        extraSource << i << "\n";
    }
	extraSource << "#define MAX_N 4"
		//<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
    extraSource << "#define fragM 8"
		//<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
    extraSource << "#define frag_N 8"
		//<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
	// extraSource << "#define KERNEL_SYNCHRONIZED_TRACING "
	// 	<< (s.synchronizedThreads ? 1 : 0)
	// 	<< "\n";
	// extraSource << getDefines(s) << "\n";
	// fillExtraSourceCode(s, extraSource);
	// for (const auto& i : getIncludeFileNames(s))
	// 	extraSource << "\n#include \"" << i << "\"\n";
	// extraSource << "#define VOLUME_INTERPOLATION_T " <<
	// 	getPerThreadType(s) << "\n";
	//extraSource << "printf(\"relu1\")" << "\n";
	// extraSource << "#define VOLUME_USE_DIRECTION " << (hasDirection ? 1 : 0) << "\n";
	extraSource << "#include \"MLPFlexible.cuh\"\n";
	const auto fun0 = renderer::KernelLoader::Instance().getKernelFunction(
		kernelName, extraSource.str(), constantNames, false, false);
	if (!fun0.has_value())
		throw std::runtime_error("Unable to compile kernel");
	const auto fun = fun0.value();
	// if (auto c = getConstantDeclarationName(s); !c.empty())
	// {
	// 	CUdeviceptr ptr = fun.constant(c);
	// 	fillConstantMemory(s, ptr, stream);
	// }

	//output tensors
	int batches = weights.size(0);
  int channels = weights.size(1);
	auto output = torch::empty({ outputdim0, outputdim2},
		at::TensorOptions().dtype(scalarType).device(c10::kCUDA));

  

    dim3 gridDim;
    dim3 blockDim;

    
    blockDim.x = blockdim;//96or192, by bs=16, 384
    blockDim.y = 1;

    // gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
    //             (WMMA_M * blockDim.x / 32);
    // gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
    gridDim.x = griddim;
    gridDim.y = 1;


	//launch kernel
	int blockSize;
	fun.bestBlockSize();
	cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);


	// {
		//blockSize = 96;//96or256;   
		
	// }
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(batches, blockSize)),
		fun.minGridSize());
	dim3 virtual_size{
		static_cast<unsigned int>(batches), 1, 1 };



	bool success = RENDERER_DISPATCH_FLOATING_TYPES(scalarType, "evaluate", [&]()
		{
      //const auto accHiddenStructure = accessor< ::kernel::Tensor1Read<scalar_t>>(hiddenStructure);
			//const auto accPosition = accessor< ::kernel::Tensor2Read<scalar_t>>(positions);
      const auto accWeights = accessor< ::kernel::Tensor2RW<scalar_t>>(weights);
      //const auto accPosition = positions;
			// const auto accDirection = hasDirection
			// 	? accessor< ::kernel::Tensor2Read<scalar_t>>(direction)
			// 	: ::kernel::Tensor2Read<scalar_t>();
      const auto accInput = accessor< ::kernel::Tensor2Read<scalar_t>>(input);
      const auto accBias = accessor< ::kernel::Tensor2Read<scalar_t>>(bias);
			const auto accOutput = accessor< ::kernel::Tensor2RW<scalar_t>>(output);

      const auto accTest = accessor< ::kernel::Tensor2Read<scalar_t>>(test);
      // const auto accDirection = direction;
			// const auto accDensity = densities;
			const void *args[] = {&test, &accWeights, &accInput, &accBias, &accOutput, &batchsizeTotal, &featuresize};
			auto result = cuLaunchKernel(
				fun.fun(), gridDim.x, gridDim.y, 1, blockDim.x, blockDim.y, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return renderer::printError(result, kernelName);
			return true;
		});

	// setBoxMin(oldBoxMin);
	// setBoxMax(oldBoxMax);
	
	if (!success) throw std::runtime_error("Error during rendering!");


	cudaEventRecord(stop);
  cudaEventSynchronize(stop);


  float milliseconds = 0;

  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Time: %f ms\n", milliseconds);
  //printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(M_GLOBAL) *
                              //                   N_GLOBAL * K_GLOBAL * 2) /
                              //                  (milliseconds / 1000.)) /
                              //  1e12);

  
	return output;
}

// static void staticCudaSourcesLoaderRec(
// 	std::vector<renderer::KernelLoader::NameAndContent>& fileList,
// 	const cmrc::directory_entry& e, const cmrc::embedded_filesystem& fs,
// 	const std::string& currentPath)
// {
// 	if (e.is_file())
// 	{
// 		std::cout << "Load file " << e.filename() << std::endl;
// 		auto f = fs.open(currentPath + e.filename());
// 		std::string content(f.size(), '\0');
// 		memcpy(content.data(), f.begin(), f.size());
// 		fileList.push_back({ e.filename(), content });
// 	} else
// 	{
// 		std::cout << "Walk directory " << currentPath << std::endl;
// 		for (const auto& e2 : fs.iterate_directory(currentPath + e.filename()))
// 			staticCudaSourcesLoaderRec(fileList, e2, fs, currentPath + e.filename() + "/");
// 	}
// }

// static void staticCudaSourcesLoader(
// 	std::vector<renderer::KernelLoader::NameAndContent>& fileList)
// {
// 	cmrc::embedded_filesystem fs = cmrc::kernels::get_filesystem();
// 	for (const auto& e : fs.iterate_directory(""))
// 		staticCudaSourcesLoaderRec(fileList, e, fs, "");
// }

// std::filesystem::path getCacheDir()
// {
// 	//suffix and default (if default, it is a relative path)
// 	static const std::filesystem::path SUFFIX{ "kernel_cache" };
// #ifdef WIN32
// 	//get the path to this dll as base path
// 	char path[MAX_PATH];
// 	HMODULE hm = NULL;

// 	if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
// 		GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
// 		(LPCSTR)&getCacheDir, &hm) == 0)
// 	{
// 		int ret = GetLastError();
// 		fprintf(stderr, "GetModuleHandle failed, error = %d\n", ret);
// 		return SUFFIX;
// 	}
// 	if (GetModuleFileName(hm, path, sizeof(path)) == 0)
// 	{
// 		int ret = GetLastError();
// 		fprintf(stderr, "GetModuleFileName failed, error = %d\n", ret);
// 		return SUFFIX;
// 	}

// 	std::filesystem::path out = path;
// 	out = out.parent_path();
// 	const auto out_str = out.string();
// 	fprintf(stdout, "This DLL is located at %s, use that as cache dir\n", out_str.c_str());
// 	out /= SUFFIX;
// 	return out;
	
// #else
// 	return SUFFIX; //default
// #endif
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   renderer::KernelLoader::Instance().initCuda();
// 	renderer::KernelLoader::Instance().setCudaCacheDir(getCacheDir());
// 	renderer::KernelLoader::Instance().setCustomCudaSourcesLoader(staticCudaSourcesLoader);
// 	cuMat::Context& ctx = cuMat::Context::current();
// 	((void)ctx);
  // m.def("forward", &matmul_forward, "Matmul forward (CUDA)");
  // m.def("backward", &matmul_backward, "Matmul backward (CUDA)");
  m.def("evaluate_flexible_MLP", &evaluate_flexible_MLP, "Flexible MLP evaluate (CUDA)");
  auto cleanup_callback = []() {
		renderer::KernelLoader::Instance().cleanup();
#if RENDERER_OPENGL_SUPPORT==1
		OffscreenContext::teardown();
#endif
	};
	m.def("cleanup", cleanup_callback, py::doc("Explicit cleanup of all CUDA references"));
	m.add_object("_cleanup", py::capsule(cleanup_callback));
}