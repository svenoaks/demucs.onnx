// DemucsProcessor.mm
#import "DemucsProcessor.h"
#import "demucs_interface.h"
#import <os/log.h>
#import <atomic>

NSErrorDomain const DemucsProcessorErrorDomain = @"DemucsProcessorErrorDomain";

// C -> Obj-C progress callback bridge (implementation remains the same)
static void progressCallbackBridge(void* context, float progress, const char* message_c) {
    DemucsProgressHandler handler = (__bridge DemucsProgressHandler)context;
    if (handler) {
        @autoreleasepool {
            NSString *message = (message_c) ? [NSString stringWithUTF8String:message_c] : @"";
            dispatch_async(dispatch_get_main_queue(), ^{
                handler(progress, message);
            });
        }
    }
}

@interface DemucsProcessor () {
    std::atomic<std::atomic<bool>*> _activeCancelFlagPtr;
}
@property (atomic, readwrite) BOOL isRunning;
@end

@implementation DemucsProcessor

// --- Singleton Implementation (using 'shared') ---
+ (instancetype)shared {
    static DemucsProcessor *sharedInstance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] initInternal];
    });
    return sharedInstance;
}

// Private initializer
- (instancetype)initInternal {
    self = [super init];
    if (self) {
        _activeCancelFlagPtr.store(nullptr);
        _isRunning = NO;
    }
    return self;
}
// --- End Singleton ---


- (BOOL)processAudioWithModelPath:(NSString *)modelPath
                     inputWavPath:(NSString *)inputWavPath
                    outputDirPath:(NSString *)outputDirPath
                  progressHandler:(nullable DemucsProgressHandler)progressHandler
                            error:(NSError **)outError
{
    std::atomic<bool>* expected_null = nullptr;
    std::atomic<bool>* new_flag = new std::atomic<bool>(false);
    if (!self->_activeCancelFlagPtr.compare_exchange_strong(expected_null, new_flag)) {
        os_log_error(OS_LOG_DEFAULT, "DemucsProcessor: Operation already in progress on shared instance.");
        if (outError) {
             NSDictionary *userInfo = @{ NSLocalizedDescriptionKey: @"Another processing operation is already in progress." };
             *outError = [NSError errorWithDomain:DemucsProcessorErrorDomain code:DEMUCS_RESULT_ERROR_UNKNOWN userInfo:userInfo];
        }
        delete new_flag;
        return NO;
    }

    self.isRunning = YES;
    std::atomic<bool>* flag_ptr_for_cpp = new_flag;

    DemucsResultCode finalResultCode = DEMUCS_RESULT_ERROR_UNKNOWN;
    NSError * _Nullable processingError = nil; // Holds the error

    @try {
        // --- Input Validation ---
        if (!modelPath || !inputWavPath || !outputDirPath) {
            finalResultCode = DEMUCS_RESULT_ERROR_INVALID_ARGS;
            NSDictionary *userInfo = @{ NSLocalizedDescriptionKey: @"Input paths cannot be nil." };
            processingError = [NSError errorWithDomain:DemucsProcessorErrorDomain code:finalResultCode userInfo:userInfo];
            return NO; // Jump to finally
        }

        // --- Prepare C strings ---
        const char *model_path_c = [modelPath UTF8String];
        const char *input_wav_path_c = [inputWavPath UTF8String];
        const char *output_dir_path_c = [outputDirPath UTF8String];
         if (!model_path_c || !input_wav_path_c || !output_dir_path_c) {
              finalResultCode = DEMUCS_RESULT_ERROR_INVALID_ARGS;
              NSDictionary *userInfo = @{ NSLocalizedDescriptionKey: @"Failed path conversion." };
              processingError = [NSError errorWithDomain:DemucsProcessorErrorDomain code:finalResultCode userInfo:userInfo];
              return NO; // Jump to finally
         }

        // --- Prepare Callbacks ---
        DemucsProgressCallback_C callback_c_ptr = NULL;
        void *context_ptr = NULL;
        if (progressHandler) {
            callback_c_ptr = progressCallbackBridge;
            context_ptr = (__bridge void *)progressHandler;
        }

        // --- Call C function ---
        finalResultCode = process_demucs_onnx_c(
            model_path_c, input_wav_path_c, output_dir_path_c,
            callback_c_ptr, context_ptr,
            (void*)flag_ptr_for_cpp
        );

        // --- Check Result Code and prepare error object if needed ---
        if (finalResultCode != DEMUCS_RESULT_SUCCESS) {
             if (finalResultCode == DEMUCS_RESULT_CANCELLED) {
                  processingError = [NSError errorWithDomain:NSCocoaErrorDomain code:NSUserCancelledError userInfo:@{NSLocalizedDescriptionKey:@"Processing was cancelled."}];
             } else {
                  // *** CORRECTED ERROR MAPPING ***
                  NSString *description; // Declare description here
                  switch (finalResultCode) {
                      // Ensure all cases from the enum are handled correctly
                      case DEMUCS_RESULT_ERROR_INVALID_ARGS: description = @"Invalid arguments provided."; break;
                      case DEMUCS_RESULT_ERROR_OUTPUT_DIR: description = @"Failed to create or access output directory."; break;
                      case DEMUCS_RESULT_ERROR_AUDIO_LOAD: description = @"Failed to load input audio file."; break;
                      case DEMUCS_RESULT_ERROR_MODEL_LOAD: description = @"Failed to load ONNX model file."; break;
                      case DEMUCS_RESULT_ERROR_WRITER_OPEN: description = @"Failed to open output WAV file for writing."; break;
                      case DEMUCS_RESULT_ERROR_WRITER_FINALIZE: description = @"Failed to finalize one or more output WAV files."; break;
                      case DEMUCS_RESULT_ERROR_INFERENCE: description = @"Error occurred during model inference."; break; // Specific message
                      case DEMUCS_RESULT_ERROR_CPP_EXCEPTION: description = @"An internal C++ exception occurred."; break;
                      case DEMUCS_RESULT_ERROR_UNKNOWN: // Fallthrough intended
                      default: description = @"An unknown error occurred during processing."; break; // Default message
                  }
                  NSDictionary *userInfo = @{ NSLocalizedDescriptionKey: description };
                  processingError = [NSError errorWithDomain:DemucsProcessorErrorDomain code:finalResultCode userInfo:userInfo];
             }
             // Don't throw, just let @finally handle the error object
        }
        // If finalResultCode was SUCCESS, processingError remains nil

    } @catch (NSException *exception) {
        // --- Catch unexpected Obj-C exceptions ---
        os_log_error(OS_LOG_DEFAULT, "DemucsProcessor: Caught unexpected exception: %{public}@", exception.reason);
        if (!processingError) { // If no specific error was already set
            NSDictionary *userInfo = @{ NSLocalizedDescriptionKey: exception.reason ?: @"An unexpected Objective-C exception occurred." };
            finalResultCode = DEMUCS_RESULT_ERROR_UNKNOWN; // Assign generic error code
            processingError = [NSError errorWithDomain:DemucsProcessorErrorDomain code:finalResultCode userInfo:userInfo];
        }
    } @finally {
        // --- Cleanup Block ---
        if (processingError && outError != nil) { *outError = processingError; } // Populate output error pointer

        // Atomically reset the flag pointer and delete the flag
        std::atomic<bool>* expected_flag_ptr = flag_ptr_for_cpp;
        if (self->_activeCancelFlagPtr.compare_exchange_strong(expected_flag_ptr, nullptr)) {
            delete flag_ptr_for_cpp;
        } else { os_log_error(OS_LOG_DEFAULT, "DemucsProcessor: Cleanup state mismatch."); }
        self.isRunning = NO;
    } // End @finally

    // Return YES only if no error object was created (i.e., finalResultCode was SUCCESS)
    return (processingError == nil);
}


// --- cancelCurrentOperation and dealloc remain the same ---
- (void)cancelCurrentOperation {
    std::atomic<bool>* flag_ptr = self->_activeCancelFlagPtr.load();
    if (flag_ptr != nullptr) {
        flag_ptr->store(true, std::memory_order_relaxed);
        os_log_info(OS_LOG_DEFAULT, "DemucsProcessor: Cancellation requested for shared instance.");
    } else {
        os_log_info(OS_LOG_DEFAULT, "DemucsProcessor: No operation running on shared instance to cancel.");
    }
}

- (void)dealloc {
    std::atomic<bool>* flag_ptr = _activeCancelFlagPtr.load();
    if (flag_ptr != nullptr) {
        flag_ptr->store(true, std::memory_order_relaxed);
        os_log_fault(OS_LOG_DEFAULT, "DemucsProcessor singleton deallocated, potentially during processing. Attempting cleanup.");
        delete flag_ptr;
    }
}

@end
