"""
Kumiko Manga/Comics Panel Extractor (WebUI)
Copyright (C) 2025 avan

This program is a web interface for the Kumiko library.
The core logic is based on Kumiko, the Comics Cutter.
Copyright (C) 2018 njean42

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import gradio as gr
import os
import tempfile
import shutil
import numpy as np
import cv2 as cv # The project uses 'cv' as an alias for cv2

# Import Kumiko's core library and its page module dependency
import kumikolib
import lib.page
from manga_panel_processor import remove_border


# ----------------------------------------------------------------------
# Core functions to solve the non-English path issue
# ----------------------------------------------------------------------

def imread_unicode(filename, flags=cv.IMREAD_COLOR):
    """
    Replaces cv.imread to support non-ASCII paths.
    """
    try:
        with open(filename, 'rb') as f:
            n = np.frombuffer(f.read(), np.uint8)
            img = cv.imdecode(n, flags)
            return img
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def imwrite_unicode(filename, img):
    """
    Replaces cv.imwrite to support non-ASCII paths.
    """
    try:
        ext = os.path.splitext(filename)[1]
        if not ext:
            ext = ".jpg" # Default to jpg if no extension
        result, n = cv.imencode(ext, img)
        if result:
            with open(filename, 'wb') as f:
                f.write(n)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error writing to file {filename}: {e}")
        return False

# ----------------------------------------------------------------------
# Monkey Patching
# This dynamically replaces the problematic functions in the original
# libraries without modifying their source code files.
# ----------------------------------------------------------------------

# Replace the cv.imread used in page.py with our new version
lib.page.cv.imread = imread_unicode

# Replace the cv.imwrite used in kumikolib.py with our new version
kumikolib.cv.imwrite = imwrite_unicode


# ----------------------------------------------------------------------
# Gradio Processing Function
# ----------------------------------------------------------------------

def process_manga_images(files, output_structure, use_rtl, remove_borders, progress=gr.Progress(track_tqdm=True)):
    """
    The main processing logic for the Gradio interface.
    Receives uploaded files and settings, processes them, and returns a path to a ZIP file.
    """
    if not files:
        raise gr.Error("Please upload at least one image file.")

    # Create temporary directories for processing
    # 1. To store the cropped panel images
    # 2. To store the final ZIP archive
    panel_output_dir = tempfile.mkdtemp(prefix="kumiko_panels_")
    zip_output_dir = tempfile.mkdtemp(prefix="kumiko_zip_")
    
    try:
        # The 'files' object from gr.Files is a list of temporary file objects
        image_paths = [file.name for file in files]
        
        progress(0, desc="Initializing Kumiko...")
        
        # Initialize Kumiko with the rtl setting from the UI
        k = kumikolib.Kumiko({
            'debug': False,
            'progress': False,  # We use Gradio's progress bar instead
            'rtl': use_rtl, # Use the value from the checkbox
            'panel_expansion': True,
        })
        
        # 1. Analyze all images
        total_files = len(image_paths)
        for i, path in enumerate(image_paths):
            progress((i + 1) / total_files, desc=f"Analyzing: {os.path.basename(path)}")
            try:
                k.parse_image(path)
            except lib.page.NotAnImageException as e:
                print(f"Warning: Skipping file {os.path.basename(path)} because it is not a valid image. Error: {e}")
                continue

        # 2. Save the panels based on the selected output structure
        #    This section replaces the original `k.save_panels()` call.
        progress(0.8, desc="Saving all panels...")
        nb_written_panels = 0
        for page in k.page_list:
            original_filename_base = os.path.splitext(os.path.basename(page.filename))[0]

            for i, panel in enumerate(page.panels):
                x, y, width, height = panel.to_xywh()
                panel_img = page.img[y:y + height, x:x + width]
                
                # If the user checked the box, attempt to remove borders
                if remove_borders:
                    panel_img = remove_border(panel_img)

                output_filepath = ""
                # Check user's choice for the output structure
                if output_structure == "Group panels in folders":
                    # Default behavior: one folder per image
                    image_specific_dir = os.path.join(panel_output_dir, original_filename_base)
                    os.makedirs(image_specific_dir, exist_ok=True)
                    output_filename = f"panel_{i}.jpg"
                    output_filepath = os.path.join(image_specific_dir, output_filename)
                else: # "Create a flat directory"
                    # New behavior: flat structure with prefixed filenames
                    output_filename = f"{original_filename_base}_panel_{i}.jpg"
                    output_filepath = os.path.join(panel_output_dir, output_filename)

                # Save the panel using our unicode-safe writer
                if imwrite_unicode(output_filepath, panel_img):
                    nb_written_panels += 1
                else:
                    print(f"\n[ERROR] Failed to write panel image to {output_filepath}\n")

        # 3. Package all cropped panels into a ZIP file
        progress(0.9, desc="Creating ZIP archive...")
        if nb_written_panels == 0:
             raise gr.Error("Analysis complete, but no croppable panels were detected.")

        zip_filename_base = os.path.join(zip_output_dir, "kumiko_output")
        zip_filepath = shutil.make_archive(zip_filename_base, 'zip', panel_output_dir)
        
        progress(1, desc="Done!")
        
        return zip_filepath

    except Exception as e:
        # Catch any other potential errors during processing
        raise gr.Error(f"An error occurred during processing: {e}")
    finally:
        # 4. Clean up the temporary directory for panels, regardless of success or failure
        shutil.rmtree(panel_output_dir)
        # Note: Gradio will automatically handle the cleanup of zip_output_dir because a file from it is returned.

# ----------------------------------------------------------------------
# Create and Launch the Gradio Interface
# ----------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Kumiko Manga/Comics Panel Extractor (WebUI)  
        Upload your manga or comic book images. This tool will automatically analyze the panels on each page,
        crop them into individual image files, and package them into a single ZIP file for you to download.
        
        This application is licensed under the **GNU Affero General Public License v3.0**.  
        The core panel detection is powered by the **kumikolib** library, created by **njean42** ([Original Project](https://github.com/njean42/kumiko)).
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Files(
                label="Upload Manga Images",
                file_count="multiple",
                file_types=["image"],
            )
            
            with gr.Accordion("Advanced Settings", open=True):
                # Add the Radio button for selecting the output structure
                output_structure_choice = gr.Radio(
                    label="ZIP File Structure",
                    choices=["Group panels in folders", "Create a flat directory"],
                    value="Group panels in folders", # Default value
                    info="Choose how to organize panels in the output ZIP file."
                )
                
                # Add the Checkbox for RTL setting
                rtl_checkbox = gr.Checkbox(
                    label="Right-to-Left (RTL) Reading Order",
                    value=True, # Default to True
                    info="Check this for manga that is read from right to left."
                )

                # Add the Checkbox for removing borders
                remove_borders_checkbox = gr.Checkbox(
                    label="Attempt to remove panel borders",
                    value=False,
                    info="Crops the image to the content area. May not be perfect for all images."
                )
            
            process_button = gr.Button("Start Analysis & Cropping", variant="primary")
        
        with gr.Column(scale=1):
            output_zip = gr.File(
                label="Download Cropped Panels (ZIP)",
            )

    process_button.click(
        fn=process_manga_images,
        inputs=[image_input, output_structure_choice, rtl_checkbox, remove_borders_checkbox],
        outputs=output_zip,
        api_name="process"
    )

    # gr.Examples(
    #     examples=[
    #         [
    #             [os.path.join(os.path.dirname(__file__), "example1.jpg"), os.path.join(os.path.dirname(__file__), "example2.png")]
    #         ]
    #     ],
    #     inputs=image_input,
    #     outputs=output_zip,
    #     fn=process_manga_images,
    #     label="Examples (place example1.jpg and example2.png in the same directory as this script)"
    # )


if __name__ == "__main__":
    demo.launch(inbrowser=True)