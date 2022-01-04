from pathlib import Path

from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

cfd = Path(__file__).parent.as_posix()


@app.route('/assets/<path:path>')
def send_js(path):
    return send_from_directory('assets', path)


@app.route('/')
def index():
    return render_template('index.html')

