#ifndef _H_cytnx_error_
#define _H_cytnx_error_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdarg.h>

#include <iostream>
#include <stdexcept>

#ifdef _MSC_VER
  #define __PRETTY_FUNCTION__ __FUNCTION__
#endif

#define cytnx_error_msg(is_true, format, ...) \
  { error_msg(__PRETTY_FUNCTION__, __FILE__, __LINE__, (is_true), (format), __VA_ARGS__); }
static inline void error_msg(char const *const func, const char *const file, int const line,
                             bool is_true, char const *format, ...) {
  try {
    if (is_true) {
      va_list args;
      char output_str[1024];
      char msg[512];
      va_start(args, format);
      vsprintf(msg, format, args);
      sprintf(output_str, "\n# Tor10 error occur at %s\n# error: %s\n# file : %s (%d)", func, msg,
              file, line);
      va_end(args);
      throw std::logic_error(output_str);
    }
  } catch (const char *output_msg) {
    std::cerr << output_msg << std::endl;
  }
}
#define cytnx_warning_msg(is_true, format, ...) \
  { warning_msg(__PRETTY_FUNCTION__, __FILE__, __LINE__, (is_true), (format), __VA_ARGS__); }
static inline void warning_msg(char const *const func, const char *const file, int const line,
                               bool is_true, char const *format, ...) {
  if (is_true) {
    va_list args;
    char output_str[1024];
    char msg[512];
    va_start(args, format);
    vsprintf(msg, format, args);
    sprintf(output_str, "\n# Tor10 warning occur at %s\n# warning: %s\n# file : %s (%d)", func, msg,
            file, line);
    va_end(args);
    std::cerr << output_str << std::endl;
  }
}

#if defined(UNI_GPU)

  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #include <cusolverDn.h>
  #include <cuComplex.h>
  #include <curand.h>

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

#endif
