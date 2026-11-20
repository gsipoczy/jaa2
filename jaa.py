##############################################################################################################
#
#                          Job Application Assistant 2 (JAA2)
#
##############################################################################################################

import gradio as gr
from utilities.execution import execute

import gradio as gr

import queue
import sys
import threading
import time
import gradio as gr


# Wrapper that captures print() and streams it to Gradio
def execute_with_live_logs(target_company, job_description, last_minute):
  log_queue = queue.Queue()

  class QueueLogger:

    def write(self, message):
      if message:
        log_queue.put(message)

    def flush(self):
      pass

  def worker():
    old_stdout = sys.stdout
    sys.stdout = QueueLogger()
    try:
      execute(target_company, job_description, last_minute)
    finally:
      sys.stdout = old_stdout
      log_queue.put(None)  # Signal that execution is complete

  threading.Thread(target=worker, daemon=True).start()

  full_log = ""
  while True:
    msg = log_queue.get()
    if msg is None:
      break
    full_log += msg
    yield full_log



with gr.Blocks(title="Job Application Assistant 2") as demo:
  gr.Markdown("# Job Application Assistant 2")

  # Target company
  target_company = gr.Textbox(
      label="Target Company",
      placeholder=("Leave empty if the Job Description's first line is the company name"),
      lines=1,
  )

  # Job Description
  job_description = gr.Textbox(
      label="Job Description",
      placeholder=("Obligatory"),
      lines=5,
  )

  # Last Minute System Prompt Extension
  last_minute = gr.Textbox(
      label="Last Minute System Prompt Extension",
      placeholder=("This text will be added to every agent's system prompt as important instruction"),
      lines=5,
  )

  # Submit button
  submit_btn = gr.Button("Submit", variant="primary")

  # Messages
  messages = gr.Textbox(label="Messages", lines=10, max_lines=10, interactive=False, autoscroll=True)

  # Your click event
  submit_btn.click(
      fn=execute_with_live_logs, inputs=[target_company, job_description, last_minute], outputs=[messages]
  )

if __name__ == "__main__":
  demo.launch()

