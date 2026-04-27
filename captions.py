import os
import json
import subprocess
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont

WORDS_PER_CAP = 4
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

STYLES = {
    "1": {
        "name":          "Golden Word",
        "font_size":     72,
        "font_file":     "Optima-Medium.ttf",
        "base_color":    (255, 255, 255, 255),
        "highlight":     (255, 190, 0,   255),
        "outline_color": (0,   0,   0,   255),
        "outline":       2,
        "bg_color":      None,
        "pill":          False,
    },
    "2": {
        "name":          "Pro Bronze",
        "font_size":     82,
        "font_file":     "Poppins-Bold.ttf",
        "base_color":    (180, 120, 60,  255),
        "highlight":     (180, 120, 60,  255),
        "outline_color": (0,   0,   80,  255),
        "outline":       4,
        "bg_color":      None,
        "pill":          False,
    },
    "3": {
        "name":          "Purple Flash",
        "font_size":     76,
        "font_file":     "Oswald-Bold.ttf",
        "base_color":    (255, 255, 255, 255),
        "highlight":     (170, 50,  220, 255),
        "outline_color": (0,   0,   0,   255),
        "outline":       3,
        "bg_color":      None,
        "pill":          False,
    },
    "4": {
        "name":          "Clean Pill",
        "font_size":     72,
        "font_file":     "Oswald-Bold.ttf",
        "base_color":    (160, 160, 160, 255),
        "highlight":     (0,   0,   0,   255),
        "outline_color": None,
        "outline":       0,
        "bg_color":      (240, 240, 240, 220),
        "pill":          True,
    },
}

FONT_FALLBACKS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def get_font(style, size):
    font_file = style.get("font_file", "")
    path = os.path.join(FONTS_DIR, font_file)
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    for fallback in FONT_FALLBACKS:
        if os.path.exists(fallback):
            return ImageFont.truetype(fallback, size)
    return ImageFont.load_default()


def draw_text_with_outline(draw, pos, text, font, fill, outline_color, outline_width):
    x, y = pos
    if outline_color and outline_width > 0:
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=fill)


def group_words(words, words_per_cap):
    merged = []
    i = 0
    while i < len(words):
        word = words[i]
        if i + 1 < len(words):
            next_word  = words[i + 1]
            next_text  = next_word["word"]
            if next_text in [",", "%", "k", "K"] or \
               (len(next_text) > 0 and next_text[0] in [",", "%"]):
                m = dict(word)
                m["word"] = word["word"] + next_word["word"]
                m["end"]  = next_word["end"]
                merged.append(m)
                i += 2
                continue
        merged.append(word)
        i += 1

    groups = []
    i = 0
    while i < len(merged):
        groups.append(merged[i:i + words_per_cap])
        i += words_per_cap
    return groups


def render_caption_frame(words_in_group, active_index, style, video_w, video_h):
    img  = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_size   = style["font_size"]
    base_color  = style["base_color"]
    highlight   = style["highlight"]
    outline_col = style["outline_color"]
    outline_w   = style["outline"]
    bg_color    = style["bg_color"]
    is_pill     = style["pill"]

    base_font      = get_font(style, font_size)
    highlight_font = get_font(style, font_size)

    word_sizes = []
    for i, word in enumerate(words_in_group):
        font = highlight_font if i == active_index else base_font
        bbox = draw.textbbox((0, 0), word, font=font)
        word_sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))

    space_bbox  = draw.textbbox((0, 0), " ", font=base_font)
    space_width = space_bbox[2] - space_bbox[0]
    total_w     = sum(ws[0] for ws in word_sizes) + space_width * (len(words_in_group) - 1)
    line_h      = max(ws[1] for ws in word_sizes)

    max_w = int(video_w * 0.85)
    if total_w > max_w and len(words_in_group) > 1:
        split   = len(words_in_group) // 2
        row1    = words_in_group[:split]
        row2    = words_in_group[split:]
        sizes1  = word_sizes[:split]
        sizes2  = word_sizes[split:]
        total_w = max(
            sum(ws[0] for ws in sizes1) + space_width * max(len(row1) - 1, 0),
            sum(ws[0] for ws in sizes2) + space_width * max(len(row2) - 1, 0)
        )
        two_rows = True
    else:
        two_rows = False

    x_start = (video_w - total_w) // 2
    y_pos   = int(video_h * 0.78) - font_size

    if is_pill and bg_color:
        pad_x             = 40
        pad_top           = 30
        pad_bottom        = 80
        total_text_height = (line_h * 2 + 10) if two_rows else line_h
        pill_x0 = (video_w // 2) - (total_w // 2) - pad_x
        pill_y0 = y_pos - pad_top
        pill_x1 = (video_w // 2) + (total_w // 2) + pad_x
        pill_y1 = y_pos + total_text_height + pad_bottom
        draw.rounded_rectangle([pill_x0, pill_y0, pill_x1, pill_y1], radius=20, fill=bg_color)

    if two_rows:
        rows       = [row1, row2]
        rows_sizes = [sizes1, sizes2]
        offset     = 0
        for row_idx, (row, rsizes) in enumerate(zip(rows, rows_sizes)):
            if not rsizes:
                continue
            row_w = sum(ws[0] for ws in rsizes) + space_width * max(len(row) - 1, 0)
            x     = (video_w - row_w) // 2
            row_y = y_pos + row_idx * (line_h + 10)
            for i, word in enumerate(row):
                gi    = offset + i
                font  = highlight_font if gi == active_index else base_font
                color = highlight if gi == active_index else base_color
                draw_text_with_outline(draw, (x, row_y), word, font, color, outline_col, outline_w)
                x += rsizes[i][0] + space_width
            offset += len(row)
    else:
        x = x_start
        for i, word in enumerate(words_in_group):
            font  = highlight_font if i == active_index else base_font
            color = highlight if i == active_index else base_color
            draw_text_with_outline(draw, (x, y_pos), word, font, color, outline_col, outline_w)
            x += word_sizes[i][0] + space_width

    return img


def get_video_info(video_path):
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,time_base",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    parts  = result.stdout.strip().split(",")
    try:
        w         = int(parts[0])
        h         = int(parts[1])
        fps_parts = parts[2].split("/")
        fps       = int(fps_parts[0]) / int(fps_parts[1])
        tb_parts  = parts[3].split("/")
        timescale = int(tb_parts[1])
    except Exception:
        w, h, fps, timescale = 1080, 1920, 24, 12288
    return w, h, fps, timescale


def export_video_with_captions(video_path, words_path, style_key, output_path):
    style = STYLES.get(str(style_key), STYLES["1"])

    with open(words_path, "r", encoding="utf-8") as f:
        words = json.load(f)

    video_w, video_h, fps, timescale = get_video_info(video_path)

    cmd_dur = [
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result       = subprocess.run(cmd_dur, capture_output=True, text=True)
    duration     = float(result.stdout.strip())
    total_frames = int(duration * fps)

    groups   = group_words(words, WORDS_PER_CAP)
    timeline = {}

    for g, group in enumerate(groups):
        if not group:
            continue
        next_start = (
            groups[g + 1][0]["start"]
            if g + 1 < len(groups) and groups[g + 1]
            else group[-1]["end"]
        )
        for wi, word in enumerate(group):
            word_start  = word["start"]
            word_end    = group[wi + 1]["start"] if wi + 1 < len(group) else next_start
            start_frame = round(word_start * fps)
            end_frame   = round(word_end * fps)
            for f in range(start_frame, min(end_frame, total_frames)):
                timeline[f] = ([w["word"] for w in group], wi)

    temp_dir     = tempfile.mkdtemp()
    overlay_path = os.path.join(temp_dir, "overlay.mp4")

    pipe_cmd = [
        FFMPEG_BIN,
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{video_w}x{video_h}",
        "-pix_fmt", "rgba",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "png",
        "-video_track_timescale", str(timescale),
        "-y", overlay_path,
    ]

    process = subprocess.Popen(pipe_cmd, stdin=subprocess.PIPE)
    blank   = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))

    for f in range(total_frames):
        if f in timeline:
            word_list, active_idx = timeline[f]
            frame = render_caption_frame(word_list, active_idx, style, video_w, video_h)
        else:
            frame = blank
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()

    merge_cmd = [
        FFMPEG_BIN,
        "-i", video_path,
        "-i", overlay_path,
        "-filter_complex", "[0:v][1:v]overlay=0:0",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        "-y", output_path,
    ]
    subprocess.run(merge_cmd, check=True)
    shutil.rmtree(temp_dir)
