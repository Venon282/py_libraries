import http.server
import socketserver
import threading
import subprocess
import requests
import time
from contextlib import contextmanager
from pathlib import Path
import shutil

@contextmanager
def temporaryPublicServer(folder: str, port: int = 8000, timeout: int = 15):
    """
    Start a local HTTP server for 'folder' and expose it via ngrok.
    Returns a function to get public URLs for files in that folder.
    """
    # Check if ngrok is installed
    if shutil.which("ngrok") is None:
        raise RuntimeError(
            "ngrok is not installed or not in PATH. "
            "Download it from https://ngrok.com/download and ensure it is accessible from the command line."
        )


    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        raise ValueError(f"{folder} is not a valid folder")
    
    # Specify the root directory
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(folder_path), **kwargs)   

    # Start local HTTP server
    httpd = socketserver.TCPServer(("", port), CustomHandler)
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Start ngrok
    ngrok_proc = subprocess.Popen(["ngrok", "http", str(port)], stdout=subprocess.PIPE)

    # Get public URL
    public_url = None
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get("http://localhost:4040/api/tunnels")
            tunnels = resp.json().get("tunnels", [])
            if tunnels:
                public_url = tunnels[0]["public_url"]
                break
        except requests.exceptions.ConnectionError:
            # ngrok API not yet up
            pass
        time.sleep(0.2)

    if not public_url:
        # If timeout reached, stop everything
        ngrok_proc.terminate()
        httpd.shutdown()
        raise RuntimeError("ngrok tunnel did not come up in time")

    def get_public_url(filename: str):
        file_path = folder_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{filename} does not exist in {folder}")
        return f"{public_url}/{filename}"

    try:
        yield get_public_url
    finally:
        # Clean up
        ngrok_proc.terminate()
        httpd.shutdown()