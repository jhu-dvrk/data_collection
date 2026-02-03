#ifndef UI_HPP
#define UI_HPP

#include "context.hpp"

// ---- Layout Constants ----
#define WINDOW_BORDER_PX 8
#define BOX_SPACING_PX 8
#define FRAME_PADDING_PX 4
#define WIDGET_SPACING_PX 4
#define WIDGET_MARGIN_PX 4
// -------------------------

void update_stage_highlighting(AppData *data);
void toggle_recording(AppData *data);
gboolean toggle_recording_idle(gpointer user_data);
void populate_audio_sources(GtkComboBoxText* combo);
void on_window_destroy_cb(GtkWidget *w, gpointer d);
void create_main_window(AppData* data);
void start_ui_update_loop(AppData* data);

#endif // UI_HPP
