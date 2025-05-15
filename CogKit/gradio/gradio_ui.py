from gradio_infer_demo import demo as demo1
from gradio_lora_demo import demo as demo2
from styles.mono import CSS, THEME

import gradio as gr


if __name__ == "__main__":
    demo = gr.TabbedInterface([demo1, demo2], ["Inference", "Train"], theme=THEME, css=CSS)
    demo.launch()
