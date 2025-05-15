import gradio as gr


THEME = gr.themes.Monochrome(
    text_size=gr.themes.Size(
        lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"
    ),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """
h1{font-size: 2em}
h3{margin-top: 0}
#component-1{text-align:center}
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
.tabitem{border: 0px}
.group_padding{padding: .55em}
"""
