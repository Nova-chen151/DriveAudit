"""Shared NAVSIM request validation, trajectory formatting and local HTTP service."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from .runtime import assert_no_evaluation_leakage


def command_one_hot(command):
    names = {0: 'RIGHT', 1: 'LEFT', 2: 'STRAIGHT'}
    code = command['code']
    if isinstance(code, bool) or code not in names or command['name'] != names[code]:
        raise ValueError('DriveAudit command code/name mismatch')
    return {'LEFT': [1, 0, 0, 0], 'STRAIGHT': [0, 1, 0, 0], 'RIGHT': [0, 0, 1, 0]}[names[code]]


def scoring_prefix(trajectory):
    points = np.asarray(trajectory, dtype=float)
    if points.shape != (8, 3) or not np.isfinite(points).all():
        raise ValueError('Expected finite native [8,3] absolute ego trajectory')
    return points[:6, :2].tolist()


def validate_model_input(value):
    allowed = {'camera_images', 'camera_calibration', 'ego_state', 'history_ego_states', 'command'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError('Only model_input fields are accepted')
    assert_no_evaluation_leakage(value)
    for name in allowed - {'history_ego_states'}:
        if name not in value: raise ValueError('Missing model input: ' + name)
    command_one_hot(value['command'])
    for name in ('velocity', 'acceleration'):
        vec = np.asarray(value['ego_state'][name], dtype=float)
        if vec.shape != (3,) or not np.isfinite(vec).all(): raise ValueError('Invalid ego ' + name)


def serve(adapter, port, **health_fields):
    class Handler(BaseHTTPRequestHandler):
        result = None

        def respond(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/health': self.respond(200, {'ready': True, 'tensors': adapter.tensor_count, **health_fields})
            elif self.path == '/driver-clean/': Handler.result = None; self.respond(200, {'reset': True})
            elif self.path == '/driver-get/': self.respond(200 if Handler.result is not None else 409, Handler.result)
            else: self.respond(404, {'error': 'unknown endpoint'})

        def do_POST(self):
            if self.path != '/driver-api/': self.respond(404, {'error': 'unknown endpoint'}); return
            Handler.result = None
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2 * 1024 * 1024: raise ValueError('Invalid request size')
                value = json.loads(self.rfile.read(size))
                if not isinstance(value, dict) or set(value) != {'model_input'}: raise ValueError('Expected model_input envelope')
                Handler.result = adapter.predict(value['model_input'])
                self.respond(201, {'ready': True})
            except (ValueError, KeyError, OSError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception as exc:
                self.respond(500, {'error': type(exc).__name__ + ': ' + str(exc)})

    HTTPServer(('127.0.0.1', port), Handler).serve_forever()
