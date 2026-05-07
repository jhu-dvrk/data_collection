#ifndef COMMON_TYPES_HPP
#define COMMON_TYPES_HPP

struct FrameData {
    long long cpu_ts;
    long long buffer_ts_ns;
    long long mono_source_ts;
    long long left_source_ts;
    long long right_source_ts;
    long long stereo_output_ts;
    long long overlay_output_ts;
};

#endif // COMMON_TYPES_HPP
