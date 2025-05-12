#include "wav_writer.hpp"
#include <iostream>
#include <vector>
#include <stdexcept> // For std::runtime_error if needed for buffer allocation
#include <cstring>   // For memset

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"


StreamingWavWriter::StreamingWavWriter()
    : wav_handle(nullptr), sample_rate(0), num_channels(0),
      total_frames_written_counter(0), is_open(false)
{
    // Allocate drwav struct on the heap
    wav_handle = new drwav();
    if (wav_handle) {
        memset(wav_handle, 0, sizeof(drwav)); // Zero initialize
    } else {
        // This is unlikely but handle allocation failure
        std::cerr << "WAV Writer Error: Failed to allocate drwav handle." << std::endl;
        // The object is in a bad state, but destructor will handle null wav_handle
    }
}

StreamingWavWriter::~StreamingWavWriter() {
    if (is_open) {
        finalize(); // Attempt to finalize if not already done
    }
    // Clean up the allocated drwav struct
    delete wav_handle;
    wav_handle = nullptr;
}

bool StreamingWavWriter::open(const std::string& filename, uint32_t sr, uint16_t nc) {
    if (!wav_handle) {
        std::cerr << "WAV Writer Error: drwav handle is null. Cannot open." << std::endl;
        return false;
    }
    if (is_open) {
        std::cerr << "WAV Writer Error: File already open: " << current_filename << std::endl;
        return false;
    }
     // dr_wav supports more channels, but let's keep the original check for now
     // if the rest of the pipeline depends on it. Remove if unnecessary.
    if (nc != 1 && nc != 2) {
         std::cerr << "WAV Writer Warning: dr_wav supports more, but keeping original 1/2 channel check. Requested: " << nc << std::endl;
         // return false; // Or allow more channels if the pipeline supports it
    }
    if (sr == 0) {
        std::cerr << "WAV Writer Error: Sample rate cannot be zero." << std::endl;
        return false;
    }

    sample_rate = sr;
    num_channels = nc;
    current_filename = filename;
    total_frames_written_counter = 0; // Reset counter

    drwav_data_format format;
    format.container = drwav_container_riff; // Standard WAV
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT; // Using 32-bit float
    format.channels = nc;
    format.sampleRate = sr;
    format.bitsPerSample = 32; // sizeof(float) * 8

    // Use drwav_init_file_write (non-sequential, requires seeking for finalize)
    // Pass NULL for allocation callbacks to use defaults (malloc/free)
    if (!drwav_init_file_write(wav_handle, filename.c_str(), &format, NULL)) {
        std::cerr << "WAV Writer Error: drwav_init_file_write failed for: " << filename << std::endl;
        is_open = false;
        // wav_handle might be in an indeterminate state, but drwav_uninit handles this
    } else {
        is_open = true;
        std::cout << "WAV Writer Info: Opened file using dr_wav: " << filename << std::endl;
    }

    return is_open;
}


bool StreamingWavWriter::append_samples(const float* interleaved_data, size_t num_total_samples) {
     if (!is_open || !wav_handle) {
         std::cerr << "WAV Writer Error: Cannot append samples, writer not open." << std::endl;
         return false;
     }
     if (num_total_samples == 0) return true; // Nothing to write

     if (num_channels == 0) {
         std::cerr << "WAV Writer Error: Cannot append samples, channel count is zero." << std::endl;
         return false;
     }
     if (num_total_samples % num_channels != 0) {
          std::cerr << "WAV Writer Error: Total number of samples (" << num_total_samples
                    << ") is not divisible by channel count (" << num_channels << ")." << std::endl;
          return false;
     }

     // dr_wav takes the number of PCM frames (samples per channel)
     drwav_uint64 num_frames_to_write = num_total_samples / num_channels;

     if (num_frames_to_write == 0) return true; // Possible if num_total_samples < num_channels

     drwav_uint64 frames_actually_written = drwav_write_pcm_frames(wav_handle, num_frames_to_write, interleaved_data);

     if (frames_actually_written != num_frames_to_write) {
          std::cerr << "WAV Writer Error: drwav_write_pcm_frames failed or wrote partial data for: "
                    << current_filename << ". Expected " << num_frames_to_write << ", wrote " << frames_actually_written << std::endl;
          // Consider the stream corrupted? Or just log? Let's log and continue for now.
          // is_open = false; // Optionally mark as failed
          total_frames_written_counter += frames_actually_written; // Still update with what was written
          return false; // Indicate error
     }

     total_frames_written_counter += frames_actually_written;
     return true;
}


bool StreamingWavWriter::append_samples(const float* chan0_data, const float* chan1_data, size_t num_samples_per_channel) {
    if (!is_open || !wav_handle) {
        std::cerr << "WAV Writer Error: Cannot append samples, writer not open." << std::endl;
        return false;
    }
    if (num_samples_per_channel == 0) return true;
    if (num_channels != 2) {
         std::cerr << "WAV Writer Error: append_samples(chan0, chan1, ...) called on non-stereo writer (channels=" << num_channels << ")." << std::endl;
         return false;
    }

    // Interleave data into the temporary buffer before writing
    size_t required_buffer_size = num_samples_per_channel * num_channels; // Total float samples
    try {
        if (interleave_buffer.size() < required_buffer_size) {
            interleave_buffer.resize(required_buffer_size);
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "WAV Writer Error: Failed to allocate interleave buffer: " << e.what() << std::endl;
        return false;
    } catch (...) {
         std::cerr << "WAV Writer Error: Unknown error allocating interleave buffer." << std::endl;
         return false;
    }


    for (size_t i = 0; i < num_samples_per_channel; ++i) {
        interleave_buffer[i * 2]     = chan0_data[i];
        interleave_buffer[i * 2 + 1] = chan1_data[i];
    }

    // Now call the interleaved version using the buffer
    // Note: drwav_write_pcm_frames takes frames, so pass num_samples_per_channel
    drwav_uint64 frames_actually_written = drwav_write_pcm_frames(wav_handle, num_samples_per_channel, interleave_buffer.data());

     if (frames_actually_written != num_samples_per_channel) {
          std::cerr << "WAV Writer Error: drwav_write_pcm_frames failed or wrote partial data for: "
                    << current_filename << ". Expected " << num_samples_per_channel << ", wrote " << frames_actually_written << std::endl;
          total_frames_written_counter += frames_actually_written;
          return false; // Indicate error
     }

     total_frames_written_counter += frames_actually_written;
     return true;
}


bool StreamingWavWriter::finalize() {
    if (!is_open) {
        // Allow finalize to be called multiple times if already closed successfully.
        // If it failed previously, is_open would be false.
        return !is_open; // Return true if it's already closed (implies success or prior failure handled)
    }
    if (!wav_handle) {
        std::cerr << "WAV Writer Error: Cannot finalize, drwav handle is null." << std::endl;
        return false; // Should not happen if is_open is true, but check anyway
    }

    drwav_result result = drwav_uninit(wav_handle);
    is_open = false; // Mark as closed regardless of result

    if (result != DRWAV_SUCCESS) {
        std::cerr << "WAV Writer Error: drwav_uninit failed for " << current_filename << " (Error code: " << result << ")" << std::endl;
        // Reset internal state? drwav_uninit should clean up the handle mostly.
        // We might want to zero out the handle again after deletion in the destructor.
        return false;
    }

    std::cout << "Finalized WAV file using dr_wav: " << current_filename << " (" << total_frames_written_counter << " frames)" << std::endl;
    return true;
}

size_t StreamingWavWriter::get_total_samples_written() const {
    // Return the number of frames (samples per channel) tracked internally
    return static_cast<size_t>(total_frames_written_counter);
}
