// demucs_interface.h
#ifndef DEMUCS_INTERFACE_H
#define DEMUCS_INTERFACE_H

#include "demucs_result_codes.h"

#ifdef __cplusplus
extern "C" {
#endif


// C-style Progress Callback Function Pointer Type
typedef void (*DemucsProgressCallback_C)(void* context, float progress, const char* message);

// Public C API Function Declaration
DemucsResultCode process_demucs_onnx_c(
    const char* model_path,
    const char* input_wav_path,
    const char* output_dir_path,
    DemucsProgressCallback_C progress_callback,
    void* progress_callback_context,
    void* cancel_flag_context // Changed to void* for opaque pointer
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // DEMUCS_INTERFACE_H
