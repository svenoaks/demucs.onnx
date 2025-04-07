// DemucsProcessor.h
#import <Foundation/Foundation.h>
#import "demucs_interface.h" // For DemucsResultCode enum

NS_ASSUME_NONNULL_BEGIN

// Progress Handler Block (Swift: @escaping (Float, String) -> Void)
typedef void (^DemucsProgressHandler)(float progress, NSString *message);

// Define NSError domain
extern NSErrorDomain const DemucsProcessorErrorDomain;

// Singleton Wrapper class
@interface DemucsProcessor : NSObject

// --- Singleton Access ---
// Provides the single, shared instance of the processor. (Reverted to 'shared')
@property (class, nonatomic, readonly, strong) DemucsProcessor *shared;

// --- Prevent Manual Instantiation ---
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

// --- Properties & Methods ---

// Indicates if a processing task is currently active on the shared instance.
@property (atomic, readonly) BOOL isRunning;

// Method to start processing using the shared instance. **Blocks the calling thread.**
// In Objective-C: Returns YES on success, NO on failure/cancel, populates error.
// In Swift: Returns Void on success, throws an Error on failure/cancel.
- (BOOL)processAudioWithModelPath:(NSString *)modelPath
                     inputWavPath:(NSString *)inputWavPath
                    outputDirPath:(NSString *)outputDirPath
                  progressHandler:(nullable DemucsProgressHandler)progressHandler
                            error:(NSError **)error
NS_SWIFT_NAME(processAudio(modelPath:inputPath:outputPath:progressHandler:));

// Method to request cancellation of the *current* operation (if running).
- (void)cancelCurrentOperation;

@end

NS_ASSUME_NONNULL_END
