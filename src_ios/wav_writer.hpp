#ifndef WAV_WRITER_HPP
#define WAV_WRITER_HPP

#include <string>
#include <vector>
#include <cstdint> // For uint16_t, uint32_t, uint64_t

// Include dr_wav.h directly to get the type definitions consistently.
// Ensure DR_WAV_IMPLEMENTATION is NOT defined here.
#include "dr_wav.h"

// No need for forward declaration 'struct drwav;' anymore

class StreamingWavWriter {
public:
    StreamingWavWriter();
    ~StreamingWavWriter(); // Ensure file is finalized even if not explicitly called

    // Opens the file using dr_wav
    // Supports only 32-bit float format internally now.
    bool open(const std::string& filename, uint32_t sample_rate, uint16_t num_channels);

    // Appends interleaved audio data (float) using dr_wav
    // num_total_samples is the TOTAL number of float samples (frames * channels)
    bool append_samples(const float* interleaved_data, size_t num_total_samples);

    // Appends separate channel data (float) (will interleave internally before calling dr_wav)
    bool append_samples(const float* chan0_data, const float* chan1_data, size_t num_samples_per_channel);

    // Finalizes the WAV file using dr_wav
    bool finalize();

    // Returns the total number of frames (samples per channel) written so far
    size_t get_total_samples_written() const;


private:
    // dr_wav handle - allocated on the heap
    // Now uses the type 'drwav' directly as defined by the included header.
    drwav* wav_handle;

    // Keep track of format for internal logic and get_total_samples_written
    uint32_t sample_rate;
    uint16_t num_channels;
    uint64_t total_frames_written_counter; // Track frames written during streaming

    // State
    bool is_open;
    std::string current_filename;

    // Temporary buffer for interleaving non-interleaved input
    std::vector<float> interleave_buffer;
};

#endif // WAV_WRITER_HPP
