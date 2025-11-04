from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
output_path = "manifesto.png"  # <-- change this path
width, height = 1200, 1600  # poster dimensions in pixels
bg_color = "white"
text_color = "black"
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf" # <-- or any elegant font you have
font_size = 36

# --- Manifesto text ---
manifesto = """🌟 MY SRK-INSPIRED MANIFESTO 🌟

I may not be born into fame, but I carry a fire within me —
the same kind that turns struggle into story and dreams into destiny.

I believe in myself, even when no one else does.
Because belief is not arrogance — it’s faith in the person I am becoming.

I will master my craft — quietly, consistently, and passionately.
No spotlight defines me; my dedication does.

I will treat every person I meet with warmth and respect.
Greatness is not about being above others,
it’s about lifting others up while I rise.

I will speak with intelligence, listen with empathy,
and act with integrity — even when no one is watching.

Confidence will be my armor, but humility will be my crown.
I will celebrate what I achieve,
but I will never forget where I started.

I don’t chase fame; I chase impact.
My “Burj Khalifa moment” may not light up a skyline,
but it will shine through the lives I touch.

And one day, when I look back,
I’ll know I didn’t just admire greatness —
I became my own version of it.
"""

# Import required modules
import textwrap

# --- Create image ---
img = Image.new("RGB", (width, height), color=bg_color)
draw = ImageDraw.Draw(img)

# Load font
font = ImageFont.truetype(font_path, font_size)

# Process text while preserving line breaks
lines = manifesto.split('\n')
wrapped_lines = []
for line in lines:
    # Only wrap lines that are too long
    if line.strip():  # If line is not empty
        wrapped = textwrap.fill(line, width=50)
        wrapped_lines.extend(wrapped.split('\n'))
    else:
        wrapped_lines.append('')  # Preserve empty lines

wrapped_text = '\n'.join(wrapped_lines)

# Calculate text position to center it
bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
x = (width - text_w) / 2
y = (height - text_h) / 2

# Draw text
draw.multiline_text((x, y), wrapped_text, fill=text_color, font=font, align="center", spacing=10)  # Added line spacing

# Save
img.save(output_path)
print(f"Poster saved successfully at {output_path}")
