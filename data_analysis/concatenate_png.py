from PIL import Image, ImageDraw
import glob
import os
import math

def pngs_to_pdf_no_resize(png_folder, output_pdf, rows=10, cols=12, dpi=600):
    png_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))
    if not png_files:
        print("No PNG files found!")
        return

    num_images = len(png_files)
    images_per_page = rows * cols
    pages = math.ceil(num_images / images_per_page)

    # Find the maximum width/height of any image:
    max_w, max_h = 0, 0
    for f in png_files:
        im = Image.open(f)
        w, h = im.size
        if w > max_w: max_w = w
        if h > max_h: max_h = h

    # Page size to fit the grid of images
    page_width = max_w * cols
    page_height = max_h * rows

    all_pages = []
    for page_idx in range(pages):
        page_images = png_files[page_idx*images_per_page : (page_idx+1)*images_per_page]

        # Make a blank page
        page_image = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(page_image)
        print("page = ", page_idx)
        for i, img_path in enumerate(page_images):
            print("image = ", i)

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            row = i // cols
            col = i % cols

            x = col * max_w
            y = row * max_h

            # Paste the image at its native size.
            # If you want them centered, you'd have to offset by (max_w-w)//2, etc.
            page_image.paste(img, (x, y))

        all_pages.append(page_image)

    # Save multi-page PDF at high DPI
    all_pages[0].save(
        output_pdf,
        save_all=True,
        append_images=all_pages[1:], 
        resolution=dpi
    )
    print(f"PDF saved at {output_pdf} with {dpi} DPI and no resizing.")

# Example Usage
png_folder = "/Users/sophiapouya/workspace/utsw/research_project/channel_data/HIP_plots_eci/"
output_pdf = "/Users/sophiapouya/workspace/utsw/research_project/analysis_plots/HIP/HIP_neurons_stretched_eci.pdf"

pngs_to_pdf_no_resize(png_folder=png_folder, output_pdf=output_pdf)