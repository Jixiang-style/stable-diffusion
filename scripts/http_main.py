import logging
import socketserver
import threading
from logging.handlers import RotatingFileHandler


def init_log():
    log_file = 'sd-log.log'
    logging.basicConfig(handlers=[RotatingFileHandler(filename=log_file, encoding='utf-8', mode='a',
                                                      maxBytes=52428800, backupCount=10),
                                  logging.StreamHandler()],
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("zeep").setLevel(logging.WARNING)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


if __name__ == '__main__':
    print(1)
    init_log()
    HOST_NAME = '0.0.0.0'
    PORT = 8080
    logging.info("serving at port: %s", PORT)
    import api.http_handler

    with ThreadedHTTPServer((HOST_NAME, PORT), api.http_handler.SDHttpHandler) as httpd:
        logging.info("serving at port: %s", PORT)
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.start()
        server_thread.join()
