#ifndef CPU_TIMESTAMP_META_HPP
#define CPU_TIMESTAMP_META_HPP

#include <gst/gst.h>
#include <ctime>

#define DC_CPU_TIMESTAMP_META_NAME "DataCollectionCPUTimestamp"

inline const GstMetaInfo *dc_cpu_timestamp_meta_register() {
    static const GstMetaInfo *info = nullptr;
    if (g_once_init_enter(&info)) {
        const GstMetaInfo *new_info =
            gst_meta_register_custom_simple(DC_CPU_TIMESTAMP_META_NAME);
        g_once_init_leave(&info, new_info);
    }
    return info;
}

inline bool dc_buffer_add_cpu_timestamp(GstBuffer *buffer, gint64 cpu_ts_ns) {
    dc_cpu_timestamp_meta_register();
    GstCustomMeta *meta = gst_buffer_add_custom_meta(buffer, DC_CPU_TIMESTAMP_META_NAME);
    if (!meta) return false;
    GstStructure *s = gst_custom_meta_get_structure(meta);
    gst_structure_set(s, "cpu-ts", G_TYPE_INT64, cpu_ts_ns, NULL);
    return true;
}

inline gint64 dc_buffer_get_cpu_timestamp(GstBuffer *buffer) {
    GstCustomMeta *meta =
        gst_buffer_get_custom_meta(buffer, DC_CPU_TIMESTAMP_META_NAME);
    if (!meta) return 0;
    GstStructure *s = gst_custom_meta_get_structure(meta);
    gint64 value = 0;
    gst_structure_get_int64(s, "cpu-ts", &value);
    return value;
}

inline gint64 dc_clock_realtime_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<gint64>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

#endif // CPU_TIMESTAMP_META_HPP
