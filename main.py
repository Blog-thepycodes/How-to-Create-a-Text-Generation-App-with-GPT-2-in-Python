import tkinter as tk
from tkinter import scrolledtext, messagebox
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import threading
import requests
from tqdm import tqdm


# Helper function to download a file with a progress bar
def download_file(url, dest):
   try:
       response = requests.get(url, stream=True)
       response.raise_for_status()
       total_size = int(response.headers.get('content-length', 0))
       with open(dest, 'wb') as file, tqdm(
               desc=f"Downloading {os.path.basename(dest)}",
               total=total_size,
               unit='iB',
               unit_scale=True,
               unit_divisor=1024,
       ) as bar:
           for data in response.iter_content(chunk_size=1024):
               size = file.write(data)
               bar.update(size)
   except requests.exceptions.RequestException as e:
       messagebox.showerror("Download Error", f"Error downloading {os.path.basename(dest)}:\n{e}")


# Download model and tokenizer if files do not exist
def download_model_with_progress(model_name="gpt2-large"):
   model_dir = f"./{model_name}"
   model_file = os.path.join(model_dir, "pytorch_model.bin")
   config_file = os.path.join(model_dir, "config.json")
   tokenizer_file = os.path.join(model_dir, "vocab.json")
   merges_file = os.path.join(model_dir, "merges.txt")


   if all(os.path.exists(f) for f in [model_file, config_file, tokenizer_file, merges_file]):
       print("Model files already downloaded.")
   else:
       model_urls = {
           "model": f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin",
           "config": f"https://huggingface.co/{model_name}/resolve/main/config.json",
           "tokenizer": f"https://huggingface.co/{model_name}/resolve/main/vocab.json",
           "merges": f"https://huggingface.co/{model_name}/resolve/main/merges.txt"
       }
       os.makedirs(model_dir, exist_ok=True)
       for name, url in model_urls.items():
           download_file(url, os.path.join(model_dir, f"{name}.bin" if name == "model" else f"{name}.json"))


   model = GPT2LMHeadModel.from_pretrained(model_dir)
   tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
   return model, tokenizer


# Function to generate text from a prompt
def generate_text(model, tokenizer, prompt, max_length=200, top_k=50, top_p=0.95, temperature=0.7):
   input_ids = tokenizer.encode(prompt, return_tensors="pt")
   output = model.generate(
       input_ids,
       max_length=max_length,
       do_sample=True,
       top_k=top_k,
       top_p=top_p,
       temperature=temperature,
       repetition_penalty=1.2,
       no_repeat_ngram_size=2
   )
   return tokenizer.decode(output[0], skip_special_tokens=True)


# Function to handle button click and update the text box with generated text using a thread
def on_generate_click():
   prompt = prompt_entry.get().strip()
   if prompt:
       result_box.delete(1.0, tk.END)
       result_box.insert(tk.END, "Generating text, please wait...")
       # Start a new thread for text generation
       threading.Thread(target=generate_and_display_text, args=(prompt,)).start()


# Function to generate and display text
def generate_and_display_text(prompt):
   max_length = int(max_length_entry.get())
   top_k = int(top_k_entry.get())
   top_p = float(top_p_entry.get())
   temperature = float(temperature_entry.get())
   generated_text = generate_text(model, tokenizer, prompt, max_length, top_k, top_p, temperature)
   result_box.delete(1.0, tk.END)
   result_box.insert(tk.END, generated_text)


# Initialize the model and tokenizer (downloads once if needed)
model, tokenizer = download_model_with_progress("gpt2-large")


# Set up the Tkinter window
window = tk.Tk()
window.title("GPT-2 Text Generator - The Pycodes")
window.geometry("800x600")


# Label for the prompt input
prompt_label = tk.Label(window, text="Enter a prompt:")
prompt_label.pack(pady=10)


# Entry field for the prompt
prompt_entry = tk.Entry(window, width=70)
prompt_entry.pack(pady=5)


# Parameter settings
param_frame = tk.Frame(window)
param_frame.pack(pady=10)


# Max length
tk.Label(param_frame, text="Max Length:").grid(row=0, column=0)
max_length_entry = tk.Entry(param_frame, width=5)
max_length_entry.insert(0, "200")
max_length_entry.grid(row=0, column=1)


# Top K
tk.Label(param_frame, text="Word Selection Range:").grid(row=0, column=2)
top_k_entry = tk.Entry(param_frame, width=5)
top_k_entry.insert(0, "50")
top_k_entry.grid(row=0, column=3)


# Top P
tk.Label(param_frame, text="Flexibility:").grid(row=0, column=4)
top_p_entry = tk.Entry(param_frame, width=5)
top_p_entry.insert(0, "0.95")
top_p_entry.grid(row=0, column=5)


# Temperature
tk.Label(param_frame, text="Creativity Level:").grid(row=0, column=6)
temperature_entry = tk.Entry(param_frame, width=5)
temperature_entry.insert(0, "0.7")
temperature_entry.grid(row=0, column=7)


# Button to trigger text generation
generate_button = tk.Button(window, text="Generate Text", command=on_generate_click)
generate_button.pack(pady=10)


# Scrolled text box to display the generated text
result_box = scrolledtext.ScrolledText(window, height=10, wrap=tk.WORD, width=80)
result_box.pack(pady=20)


# Run the Tkinter main loop
window.mainloop()
