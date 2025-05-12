// demucs_result_codes.h
#ifndef DEMUCS_RESULT_CODES_H
#define DEMUCS_RESULT_CODES_H

// This header defines result codes shared between the C++ core and C interfaces.

// Use extern "C" guards for C++ compatibility, allowing this header
// to be included directly by C++ source files.
#ifdef __cplusplus
extern "C" {
#endif

// Result Codes Enum (Same definition as before)
typedef enum {
    DEMUCS_RESULT_SUCCESS = 0,
    DEMUCS_RESULT_ERROR_UNKNOWN = 1,
    DEMUCS_RESULT_ERROR_INVALID_ARGS = 2,
    DEMUCS_RESULT_ERROR_OUTPUT_DIR = 3,
    DEMUCS_RESULT_ERROR_AUDIO_LOAD = 4,
    DEMUCS_RESULT_ERROR_MODEL_LOAD = 5,
    DEMUCS_RESULT_ERROR_WRITER_OPEN = 6,
    DEMUCS_RESULT_ERROR_WRITER_FINALIZE = 7,
    DEMUCS_RESULT_ERROR_INFERENCE = 8,
    DEMUCS_RESULT_ERROR_CPP_EXCEPTION = 9,
    DEMUCS_RESULT_CANCELLED = 10,
    DEMUCS_RESULT_ERROR_OUT_OF_MEMORY = 11,
    DEMUCS_RESULT_ERROR_BUFFER_OVERFLOW = 12
} DemucsResultCode;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // DEMUCS_RESULT_CODES_H
