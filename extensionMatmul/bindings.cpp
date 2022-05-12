#include <torch/extension.h>

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
// CUDA forward declarations
// from .cu !!
// std::vector<torch::Tensor> matmul_cuda_forward(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias);

// std::vector<torch::Tensor> matmul_cuda_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell);

// C++ interface
// //from bindings.cpp//
// //#include <torch/extension.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include <tinyformat.h>
// #include <cuMat/src/Context.h>
// //#include <glm/gtx/string_cast.hpp>
// #include <third-party/Eigen/Core> // in cuMat

// //#include <kernel_loader.h>
// //#include <module_registry.h>
// //#include <opengl_utils.h>

// #ifdef WIN32
// #ifndef NOMINMAX
// #define NOMINMAX 1
// #endif
// #include <Windows.h>
// #endif

// #ifndef TORCH_EXTENSION_NAME
// #define TORCH_EXTENSION_NAME pyrenderer
// #endif

namespace py = pybind11;
// //using namespace renderer;

// #include "cmrc.hpp"
// CMRC_DECLARE(kernels);
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// std::vector<torch::Tensor> matmul_forward( //attention std::vector
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(bias);
 
//   return matmul_cuda_forward(input, weights, bias);
// }

// std::vector<torch::Tensor> matmul_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell) {
//   CHECK_INPUT(grad_h);
//   CHECK_INPUT(grad_cell);

//   return matmul_cuda_backward(
//       grad_h,
//       grad_cell);
// }

// std::vector<torch::Tensor> evaluate( //attention std::vector
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(bias);
 
//   return matmul_cuda_forward(input, weights, bias);
// }

torch::Tensor evaluate(
	torch::Tensor& positions, const torch::Tensor& direction, const torch::Tensor& bias)
{
  CUstream stream = c10::cuda::getCurrentCUDAStream();
	// CHECK_CUDA(positions, true);
	// CHECK_DIM(positions, 2);
	// CHECK_SIZE(positions, 1, 3);
	bool hasDirection = false;
	// if (direction.defined())
	// {
	// 	hasDirection = true;
	// 	CHECK_CUDA(direction, true);
	// 	CHECK_DIM(direction, 2);
	// 	CHECK_SIZE(direction, 1, 3);
	// }

	//GlobalSettings s{};
	auto scalarType = positions.scalar_type();
	// s.volumeShouldProvideNormals = false;
	// s.interpolationInObjectSpace = false;
	// const auto oldBoxMax = boxMax();
	// const auto oldBoxMin = boxMin();
	// setBoxMin(make_double3(0, 0, 0));
	// setBoxMax(make_double3(1, 1, 1));
	// int channels = outputChannels();

	//kernel
	// this->prepareRendering(s);
	const std::string kernelName = "EvaluateNoBatches2";
	std::vector<std::string> constantNames;
	// if (const auto c = getConstantDeclarationName(s); !c.empty())
	// 	constantNames.push_back(c);
	std::stringstream extraSource;
	// extraSource << "#define KERNEL_DOUBLE_PRECISION "
	// 	<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
	// 	<< "\n";
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
	extraSource << "#include \"relu.cuh\"\n";
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
	int batches = positions.size(0);
  int channels = positions.size(1);
	auto densities = torch::empty({ batches, channels },
		at::TensorOptions().dtype(scalarType).device(c10::kCUDA));

	//launch kernel
	int blockSize;
	// if (!1>0)//(s.fixedBlockSize>0)
	// {
	// 	if (s.fixedBlockSize > fun.bestBlockSize())
	// 		throw std::runtime_error("larger block size requested that can be fulfilled");
	// 	blockSize = s.fixedBlockSize;
	// } else
	// {
		blockSize = fun.bestBlockSize();
	// }
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(batches, blockSize)),
		fun.minGridSize());
	dim3 virtual_size{
		static_cast<unsigned int>(batches), 1, 1 };
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(scalarType, "IVolumeInterpolation::evaluate", [&]()
		{
			const auto accPosition = accessor< ::kernel::Tensor2Read<scalar_t>>(positions);
			const auto accDirection = hasDirection
				? accessor< ::kernel::Tensor2Read<scalar_t>>(direction)
				: ::kernel::Tensor2Read<scalar_t>();
			const auto accDensity = accessor< ::kernel::Tensor2RW<scalar_t>>(densities);
			const void* args[] = { &accPosition, &accDensity};
			auto result = cuLaunchKernel(
				fun.fun(), minGridSize, 1, 1, blockSize, 1, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return renderer::printError(result, kernelName);
			return true;
		});

	// setBoxMin(oldBoxMin);
	// setBoxMax(oldBoxMax);
	
	if (!success) throw std::runtime_error("Error during rendering!");
	
	return positions;
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
  m.def("evaluate", &evaluate, "Matmul evaluate (CUDA)");
  auto cleanup_callback = []() {
		renderer::KernelLoader::Instance().cleanup();
#if RENDERER_OPENGL_SUPPORT==1
		OffscreenContext::teardown();
#endif
	};
	m.def("cleanup", cleanup_callback, py::doc("Explicit cleanup of all CUDA references"));
	m.add_object("_cleanup", py::capsule(cleanup_callback));
}