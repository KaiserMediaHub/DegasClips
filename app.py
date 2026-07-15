import os
import json
import threading
import uuid

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, send_file
)
from werkzeug.utils import secure_filename

from database import init_db, get_db
import transcription
import captions
from sync_logic import update_words_from_segments

# ── Config ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key    = os.environ.get("SECRET_KEY", "change-me-in-production")
APP_PASSWORD      = os.environ.get("APP_PASSWORD", "degas2024")
UPLOAD_FOLDER     = os.environ.get("UPLOAD_FOLDER", "uploads")
OUTPUT_FOLDER     = os.environ.get("OUTPUT_FOLDER", "outputs")
MAX_CONTENT_MB    = int(os.environ.get("MAX_CONTENT_MB", "2048"))

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@app.template_filter("datestr")
def datestr_filter(v):
    return str(v)[:10] if v else ""


def
