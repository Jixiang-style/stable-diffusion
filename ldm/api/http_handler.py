import json
import logging
from http.server import BaseHTTPRequestHandler

from api.processing import Processing


class SDHttpHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        request_path = self.path.split('?')[0]
        try:
            if request_path == '/txt2img':
                self.txt2img()
            else:
                self.send_error(404)
        except Exception:
            logging.error("error occurred", exc_info=True)
            self.send_error(500)

    def txt2img(self):
        data = self.get_request_body()
        print(data)
        Processing.txt2img(data)
        logging.info("txt2img")
        json_str = json.dumps({'status': 'success'}).encode(encoding='utf_8')
        self.send_json(json_str)

    def get_request_body(self):
        req_body = self.rfile.read(int(self.headers['content-length']))
        req_data = json.loads(req_body.decode())
        return req_data

    def send_json(self, result):
        self.send_value('Content-type: application/json', result)

    def send_image(self, image):
        self.send_value('Content-type: image/png', image)

    def send_value(self, content_type, return_obj):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(return_obj)

    def log_message(self, format, *args):
        pass
