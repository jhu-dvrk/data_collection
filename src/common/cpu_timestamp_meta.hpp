#ifndef CPU_TIMESTAMP_META_HPP
#define CPU_TIMESTAMP_META_HPP

#include <gst/gst.h>
#include <ctime>

#define DC_FRAME_TIMESTAMPS_META_NAME "DataCollectionFrameTimestamps"

struct DcFrameTimestamps {
    gint64 mono_source_ts = 0;
    gint64 left_source_ts = 0;
    gint64 right_source_ts = 0;
    gint64 stereo_output_ts = 0;
    gint64 overlay_output_ts = 0;
};

inline const GstMetaInfo *dc_frame_timestamps_meta_register() {
    static const GstMetaInfo *info = nullptr;
    if (g_once_init_enter(&info)) {
        const GstMetaInfo *new_info =
            gst_meta_register_custom_simple(DC_FRAME_TIMESTAMPS_META_NAME);
        g_once_init_leave(&info, new_info);
    }
    return info;
}

inline GstStructure *dc_buffer_get_frame_timestamps_structure(GstBuffer *buffer) {
    if (!buffer) return nullptr;
    dc_frame_timestamps_meta_register();
    GstCustomMeta *meta =
        gst_buffer_get_custom_meta(buffer, DC_FRAME_TIMESTAMPS_META_NAME);
    if (!meta) {
        meta = gst_buffer_add_custom_meta(buffer, DC_FRAME_TIMESTAMPS_META_NAME);
    }
    return meta ? gst_custom_meta_get_structure(meta) : nullptr;
}

inline bool dc_buffer_set_frame_timestamps(GstBuffer *buffer,
                                           const DcFrameTimestamps &timestamps) {
    GstStructure *s = dc_buffer_get_frame_timestamps_structure(buffer);
    if (!s) return false;
    gst_structure_set(s,
                      "mono-source-ts", G_TYPE_INT64, timestamps.mono_source_ts,
                      "left-source-ts", G_TYPE_INT64, timestamps.left_source_ts,
                      "right-source-ts", G_TYPE_INT64, timestamps.right_source_ts,
                      "stereo-output-ts", G_TYPE_INT64, timestamps.stereo_output_ts,
                      "overlay-output-ts", G_TYPE_INT64, timestamps.overlay_output_ts,
                      NULL);
    return true;
}

inline DcFrameTimestamps dc_buffer_get_frame_timestamps(GstBuffer *buffer) {
    DcFrameTimestamps timestamps;
    if (!buffer) return timestamps;
    GstCustomMeta *meta =
        gst_buffer_get_custom_meta(buffer, DC_FRAME_TIMESTAMPS_META_NAME);
    if (!meta) return timestamps;

    GstStructure *s = gst_custom_meta_get_structure(meta);
    gst_structure_get_int64(s, "mono-source-ts", &timestamps.mono_source_ts);
    gst_structure_get_int64(s, "left-source-ts", &timestamps.left_source_ts);
    gst_structure_get_int64(s, "right-source-ts", &timestamps.right_source_ts);
    gst_structure_get_int64(s, "stereo-output-ts", &timestamps.stereo_output_ts);
    gst_structure_get_int64(s, "overlay-output-ts", &timestamps.overlay_output_ts);
    return timestamps;
}

inline bool dc_buffer_has_frame_timestamps(GstBuffer *buffer) {
    return buffer &&
           gst_buffer_get_custom_meta(buffer, DC_FRAME_TIMESTAMPS_META_NAME) != nullptr;
}

inline gint64 dc_clock_realtime_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<gint64>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

#endif // CPU_TIMESTAMP_META_HPP
