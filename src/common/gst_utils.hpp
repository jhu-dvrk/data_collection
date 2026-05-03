#ifndef DATA_COLLECTION_COMMON_GST_UTILS_HPP
#define DATA_COLLECTION_COMMON_GST_UTILS_HPP

#include <gst/gst.h>
#include <string>

namespace dc {

/**
 * Dump the GStreamer pipeline to a .dot file for Graphviz.
 * 
 * @param pipeline The GStreamer pipeline (GstBin)
 * @param path The full path to the output file (including .dot extension)
 * @param flags Details level (GstDebugGraphDetails)
 */
void dump_dot(GstElement* pipeline, const std::string& path, GstDebugGraphDetails flags = GST_DEBUG_GRAPH_SHOW_ALL);

/**
 * Standard command-line argument parser for dot generation flags.
 * 
 * @param i Current argument index (will be incremented if an argument is consumed)
 * @param argc Total argument count
 * @param argv Argument array
 * @param dump_dot [out] Set to true if -g/--dot is found
 * @param dot_flags [out] Updated with the selected detail level
 * @return true if the argument was handled, false otherwise
 */
bool parse_dot_arguments(int& i, int argc, char* argv[], bool& dump_dot, GstDebugGraphDetails& dot_flags);

/**
 * Print standard usage for dot generation flags.
 */
void print_dot_usage();

} // namespace dc

#endif // DATA_COLLECTION_COMMON_GST_UTILS_HPP
