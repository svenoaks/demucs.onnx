// wav_reader.hpp
#ifndef WAV_READER_HPP
#define WAV_READER_HPP

#include <string>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense> // For Eigen::MatrixXf
#include "dr_wav.h"     // dr_wav header

class StreamingWavReader {
public:
    StreamingWavReader();
    ~StreamingWavReader();

    // Delete copy constructor and assignment operator
    StreamingWavReader(const StreamingWavReader&) = delete;
    StreamingWavReader& operator=(const StreamingWavReader&) = delete;

    // Open the WAV file
    bool open(const std::string& filename);

    // Read a chunk of audio data
    // Returns the number of frames actually read.
    // Reads into the provided buffer, resizing if necessary.
    drwav_uint64 read_chunk(Eigen::MatrixXf& buffer, drwav_uint64 max_frames_to_read);

    // Seek to a specific PCM frame
    bool seek_to_frame(drwav_uint64 frame_index);

    // Get audio properties
    uint32_t get_sample_rate() const;
    uint16_t get_channels() const;
    drwav_uint64 get_total_frames() const;

    // Close the file
    void close();

    // Check if file is open
    bool is_open() const;

private:
    drwav wav;
    bool is_file_open = false;
    std::string current_filename;
    uint32_t sample_rate = 0;
    uint16_t channels = 0;
    drwav_uint64 total_pcm_frames = 0;
    std::vector<float> read_buffer_interleaved; // Internal buffer for dr_wav read
};

#endif // WAV_READER_HPP
