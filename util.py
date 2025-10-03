import io
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image
import chess

def save_game_gif(board, output_path, duration=500, loop=0):
    svgs = []

    b = chess.Board()
    for move in board.move_stack:
        b.push(move)
        svgs.append(b._repr_svg_())

    png_frames = []

    for svg_data in svgs:
        if isinstance(svg_data, str):
            drawing = svg2rlg(io.StringIO(svg_data))
        else:
            svg_data.seek(0)
            drawing = svg2rlg(svg_data)

        png_data = io.BytesIO()
        renderPM.drawToFile(drawing, png_data, fmt="PNG")
        png_data.seek(0)
        
        image = Image.open(png_data)
        png_frames.append(image)

    if png_frames:
        first_frame = png_frames[0]
        other_frames = png_frames[1:]
        first_frame.save(
            output_path,
            format="GIF",
            append_images=other_frames,
            save_all=True,
            duration=duration,
            loop=loop
        )
        print(f"GIF successfully saved to {output_path}")