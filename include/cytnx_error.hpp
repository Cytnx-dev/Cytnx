#ifndef CYTNX_CYTNX_ERROR_H_
#define CYTNX_CYTNX_ERROR_H_

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#if defined(__has_include)
  #if __has_include(<execinfo.h>)
    #include <execinfo.h>
    #define CYTNX_HAS_EXECINFO 1
  #endif
#endif

#ifndef CYTNX_HAS_EXECINFO
  #define CYTNX_HAS_EXECINFO 0
#endif

#ifdef _MSC_VER
  #define __PRETTY_FUNCTION__ __FUNCTION__
#endif

namespace cytnx_error_detail {

  static inline std::string format_message(char const *format, va_list args) {
    if (format == nullptr) {
      return {};
    }

    va_list count_args;
    va_copy(count_args, args);
    const int count = std::vsnprintf(nullptr, 0, format, count_args);
    va_end(count_args);
    if (count < 0) {
      return std::string("[formatting failed] ") + format;
    }

    std::vector<char> buffer(static_cast<std::size_t>(count) + 1);
    va_list write_args;
    va_copy(write_args, args);
    const int written = std::vsnprintf(buffer.data(), buffer.size(), format, write_args);
    va_end(write_args);
    if (written < 0) {
      return std::string("[formatting failed] ") + format;
    }
    return std::string(buffer.data(), static_cast<std::size_t>(written));
  }

  static inline std::string format_report(char const *kind, char const *func, char const *file,
                                          int line, const std::string &message) {
    std::ostringstream out;
    out << "\n# Cytnx " << kind << " occur at " << func << "\n# " << kind << ": " << message
        << "\n# file : " << file << " (" << line << ")";
    return out.str();
  }

  static inline void print_stack_trace() {
#if CYTNX_HAS_EXECINFO
    std::cerr << "Stack trace:" << std::endl;
    void *array[10];
    const std::size_t size = backtrace(array, 10);
    char **strings = backtrace_symbols(array, size);
    if (strings != nullptr) {
      for (std::size_t i = 0; i < size; i++) {
        std::cerr << strings[i] << std::endl;
      }
      free(strings);
    }
#else
    std::cerr << "Stack trace is unavailable on this platform/compiler." << std::endl;
#endif
  }

  static inline bool stack_trace_setting_enabled(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return !(value == "0" || value == "false" || value == "off" || value == "no");
  }

  static inline bool stack_trace_enabled() {
    const char *setting = std::getenv("CYTNX_STACKTRACE");
    return setting == nullptr || stack_trace_setting_enabled(std::string(setting));
  }

}  // namespace cytnx_error_detail

#define cytnx_error(format, ...) \
  { error_msg(__PRETTY_FUNCTION__, __FILE__, __LINE__, (format)__VA_OPT__(, ) __VA_ARGS__); }
#define cytnx_error_if(is_true, format, ...)                                                  \
  {                                                                                           \
    if (is_true) {                                                                            \
      error_msg(__PRETTY_FUNCTION__, __FILE__, __LINE__, (format)__VA_OPT__(, ) __VA_ARGS__); \
    }                                                                                         \
  }
#define cytnx_error_msg(is_true, format, ...) \
  cytnx_error_if(is_true, format __VA_OPT__(, ) __VA_ARGS__)
static inline void error_msg(char const *const func, const char *const file, int const line,
                             char const *format, ...) {
  va_list args;
  va_start(args, format);
  const std::string message = cytnx_error_detail::format_message(format, args);
  va_end(args);

  const std::string output_str =
    cytnx_error_detail::format_report("error", func, file, line, message);
  std::cerr << output_str << std::endl;
  if (cytnx_error_detail::stack_trace_enabled()) {
    cytnx_error_detail::print_stack_trace();
  }
  throw std::logic_error(output_str);
}
#define cytnx_warning(format, ...) \
  { warning_msg(__PRETTY_FUNCTION__, __FILE__, __LINE__, (format)__VA_OPT__(, ) __VA_ARGS__); }
#define cytnx_warning_if(is_true, format, ...)                                                  \
  {                                                                                             \
    if (is_true) {                                                                              \
      warning_msg(__PRETTY_FUNCTION__, __FILE__, __LINE__, (format)__VA_OPT__(, ) __VA_ARGS__); \
    }                                                                                           \
  }
#define cytnx_warning_msg(is_true, format, ...) \
  cytnx_warning_if(is_true, format __VA_OPT__(, ) __VA_ARGS__)
static inline void warning_msg(char const *const func, const char *const file, int const line,
                               char const *format, ...) {
  va_list args;
  va_start(args, format);
  const std::string message = cytnx_error_detail::format_message(format, args);
  va_end(args);

  const std::string output_str =
    cytnx_error_detail::format_report("warning", func, file, line, message);
  std::cerr << output_str << std::endl;
}

#if defined(UNI_GPU)

  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #include <cusolverDn.h>
  #include <cuComplex.h>
  #include <curand.h>

  #if defined(UNI_CUTENSOR)
    #include <cutensor.h>
  #endif

  #ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return "cudaSuccess";

    case cudaErrorMissingConfiguration:
      return "cudaErrorMissingConfiguration";

    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";

    case cudaErrorInitializationError:
      return "cudaErrorInitializationError";

    case cudaErrorLaunchFailure:
      return "cudaErrorLaunchFailure";

    case cudaErrorPriorLaunchFailure:
      return "cudaErrorPriorLaunchFailure";

    case cudaErrorLaunchTimeout:
      return "cudaErrorLaunchTimeout";

    case cudaErrorLaunchOutOfResources:
      return "cudaErrorLaunchOutOfResources";

    case cudaErrorInvalidDeviceFunction:
      return "cudaErrorInvalidDeviceFunction";

    case cudaErrorInvalidConfiguration:
      return "cudaErrorInvalidConfiguration";

    case cudaErrorInvalidDevice:
      return "cudaErrorInvalidDevice";

    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";

    case cudaErrorInvalidPitchValue:
      return "cudaErrorInvalidPitchValue";

    case cudaErrorInvalidSymbol:
      return "cudaErrorInvalidSymbol";

    case cudaErrorMapBufferObjectFailed:
      return "cudaErrorMapBufferObjectFailed";

    case cudaErrorUnmapBufferObjectFailed:
      return "cudaErrorUnmapBufferObjectFailed";

    case cudaErrorInvalidHostPointer:
      return "cudaErrorInvalidHostPointer";

    case cudaErrorInvalidDevicePointer:
      return "cudaErrorInvalidDevicePointer";

    case cudaErrorInvalidTexture:
      return "cudaErrorInvalidTexture";

    case cudaErrorInvalidTextureBinding:
      return "cudaErrorInvalidTextureBinding";

    case cudaErrorInvalidChannelDescriptor:
      return "cudaErrorInvalidChannelDescriptor";

    case cudaErrorInvalidMemcpyDirection:
      return "cudaErrorInvalidMemcpyDirection";

    case cudaErrorAddressOfConstant:
      return "cudaErrorAddressOfConstant";

    case cudaErrorTextureFetchFailed:
      return "cudaErrorTextureFetchFailed";

    case cudaErrorTextureNotBound:
      return "cudaErrorTextureNotBound";

    case cudaErrorSynchronizationError:
      return "cudaErrorSynchronizationError";

    case cudaErrorInvalidFilterSetting:
      return "cudaErrorInvalidFilterSetting";

    case cudaErrorInvalidNormSetting:
      return "cudaErrorInvalidNormSetting";

    case cudaErrorMixedDeviceExecution:
      return "cudaErrorMixedDeviceExecution";

    case cudaErrorCudartUnloading:
      return "cudaErrorCudartUnloading";

    case cudaErrorUnknown:
      return "cudaErrorUnknown";

    case cudaErrorNotYetImplemented:
      return "cudaErrorNotYetImplemented";

    case cudaErrorMemoryValueTooLarge:
      return "cudaErrorMemoryValueTooLarge";

    case cudaErrorInvalidResourceHandle:
      return "cudaErrorInvalidResourceHandle";

    case cudaErrorNotReady:
      return "cudaErrorNotReady";

    case cudaErrorInsufficientDriver:
      return "cudaErrorInsufficientDriver";

    case cudaErrorSetOnActiveProcess:
      return "cudaErrorSetOnActiveProcess";

    case cudaErrorInvalidSurface:
      return "cudaErrorInvalidSurface";

    case cudaErrorNoDevice:
      return "cudaErrorNoDevice";

    case cudaErrorECCUncorrectable:
      return "cudaErrorECCUncorrectable";

    case cudaErrorSharedObjectSymbolNotFound:
      return "cudaErrorSharedObjectSymbolNotFound";

    case cudaErrorSharedObjectInitFailed:
      return "cudaErrorSharedObjectInitFailed";

    case cudaErrorUnsupportedLimit:
      return "cudaErrorUnsupportedLimit";

    case cudaErrorDuplicateVariableName:
      return "cudaErrorDuplicateVariableName";

    case cudaErrorDuplicateTextureName:
      return "cudaErrorDuplicateTextureName";

    case cudaErrorDuplicateSurfaceName:
      return "cudaErrorDuplicateSurfaceName";

    case cudaErrorDevicesUnavailable:
      return "cudaErrorDevicesUnavailable";

    case cudaErrorInvalidKernelImage:
      return "cudaErrorInvalidKernelImage";

    case cudaErrorNoKernelImageForDevice:
      return "cudaErrorNoKernelImageForDevice";

    case cudaErrorIncompatibleDriverContext:
      return "cudaErrorIncompatibleDriverContext";

    case cudaErrorPeerAccessAlreadyEnabled:
      return "cudaErrorPeerAccessAlreadyEnabled";

    case cudaErrorPeerAccessNotEnabled:
      return "cudaErrorPeerAccessNotEnabled";

    case cudaErrorDeviceAlreadyInUse:
      return "cudaErrorDeviceAlreadyInUse";

    case cudaErrorProfilerDisabled:
      return "cudaErrorProfilerDisabled";

    case cudaErrorProfilerNotInitialized:
      return "cudaErrorProfilerNotInitialized";

    case cudaErrorProfilerAlreadyStarted:
      return "cudaErrorProfilerAlreadyStarted";

    case cudaErrorProfilerAlreadyStopped:
      return "cudaErrorProfilerAlreadyStopped";

    /* Since CUDA 4.0*/
    case cudaErrorAssert:
      return "cudaErrorAssert";

    case cudaErrorTooManyPeers:
      return "cudaErrorTooManyPeers";

    case cudaErrorHostMemoryAlreadyRegistered:
      return "cudaErrorHostMemoryAlreadyRegistered";

    case cudaErrorHostMemoryNotRegistered:
      return "cudaErrorHostMemoryNotRegistered";

    /* Since CUDA 5.0 */
    case cudaErrorOperatingSystem:
      return "cudaErrorOperatingSystem";

    case cudaErrorPeerAccessUnsupported:
      return "cudaErrorPeerAccessUnsupported";

    case cudaErrorLaunchMaxDepthExceeded:
      return "cudaErrorLaunchMaxDepthExceeded";

    case cudaErrorLaunchFileScopedTex:
      return "cudaErrorLaunchFileScopedTex";

    case cudaErrorLaunchFileScopedSurf:
      return "cudaErrorLaunchFileScopedSurf";

    case cudaErrorSyncDepthExceeded:
      return "cudaErrorSyncDepthExceeded";

    case cudaErrorLaunchPendingCountExceeded:
      return "cudaErrorLaunchPendingCountExceeded";

    case cudaErrorNotPermitted:
      return "cudaErrorNotPermitted";

    case cudaErrorNotSupported:
      return "cudaErrorNotSupported";

    /* Since CUDA 6.0 */
    case cudaErrorHardwareStackError:
      return "cudaErrorHardwareStackError";

    case cudaErrorIllegalInstruction:
      return "cudaErrorIllegalInstruction";

    case cudaErrorMisalignedAddress:
      return "cudaErrorMisalignedAddress";

    case cudaErrorInvalidAddressSpace:
      return "cudaErrorInvalidAddressSpace";

    case cudaErrorInvalidPc:
      return "cudaErrorInvalidPc";

    case cudaErrorIllegalAddress:
      return "cudaErrorIllegalAddress";

    /* Since CUDA 6.5*/
    case cudaErrorInvalidPtx:
      return "cudaErrorInvalidPtx";

    case cudaErrorInvalidGraphicsContext:
      return "cudaErrorInvalidGraphicsContext";

    case cudaErrorStartupFailure:
      return "cudaErrorStartupFailure";

    case cudaErrorApiFailureBase:
      return "cudaErrorApiFailureBase";

    /* Since CUDA 8.0*/
    case cudaErrorNvlinkUncorrectable:
      return "cudaErrorNvlinkUncorrectable";
    default:
      break;
  }

  return "<unknown>";
}
  #endif

  #ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}
  #endif

  #ifdef CUSOLVER_COMMON_H_
// cuSOLVER API errors
static const char *_cudaGetErrorEnum(cusolverStatus_t error) {
  switch (error) {
    case CUSOLVER_STATUS_SUCCESS:
      return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "CUSOLVER_STATUS_INVALID_LICENSE";
  }

  return "<unknown>";
}
  #endif

  #ifdef CURAND_H_
// cuRAND API errors
static const char *_cudaGetErrorEnum(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
  #endif

  #ifdef UNI_CUTENSOR
static const char *_cudaGetErrorEnum(cutensorStatus_t error) {
  return cutensorGetErrorString(error);
}
  #endif

  #ifdef __DRIVER_TYPES_H__
    #ifndef DEVICE_RESET
      #define DEVICE_RESET cudaDeviceReset();
    #endif
  #else
    #ifndef DEVICE_RESET
      #define DEVICE_RESET
    #endif
  #endif

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    DEVICE_RESET
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

  #ifdef __DRIVER_TYPES_H__
    // This will output the proper CUDA error strings in the event that a CUDA host call returns an
    // error
    #define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

  #endif

#endif  // End of #if defined(UNI_GPU)

#endif  // CYTNX_CYTNX_ERROR_H_
