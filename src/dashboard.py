import dearpygui.dearpygui as dpg

dpg.create_context()

with dpg.window(label="Dashboard"):
    dpg.add_text("Hello Dashboard")

dpg.create_viewport(
    title="Face Dashboard",
    width=1200,
    height=800
)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()

dpg.destroy_context()