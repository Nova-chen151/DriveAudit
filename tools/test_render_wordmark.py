"""Local regression tests for the source-derived logo renderer.

Run: python -I -B -m unittest discover -s tools -p 'test_render_wordmark.py'
"""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import render_wordmark as renderer


class WordmarkTests(unittest.TestCase):
    def test_gradient_is_rendered_in_color_not_black(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10" viewBox="0 0 20 10">
        <defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="20" y2="0">
        <stop offset="0" stop-color="#ff0000"/><stop offset="1" stop-color="#0000ff"/>
        </linearGradient></defs><path fill="url(#g)" d="M0,0 L20,0 L20,10 L0,10 Z"/></svg>'''
        image = Image.open(BytesIO(renderer.render_svg(svg, 20, 10, 2)))
        self.assertEqual(image.size, (40, 20))
        self.assertGreater(image.getpixel((1, 10))[0], 240)
        self.assertGreater(image.getpixel((38, 10))[2], 240)
        self.assertEqual(image.getpixel((20, 10))[3], 255)

    def test_transparency_is_straight_alpha_without_dark_fringe(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10" viewBox="0 0 20 10">
        <path fill="#ffffff" d="M2.3,2.3 L17.7,2.3 L17.7,7.7 L2.3,7.7 Z"/></svg>'''
        array = np.asarray(Image.open(BytesIO(renderer.render_svg(svg, 20, 10, 2))))
        partial = (array[..., 3] > 0) & (array[..., 3] < 255)
        self.assertTrue(partial.any())
        self.assertTrue(np.all(array[..., :3][partial] == 255))
        self.assertEqual(int(array[0, 0, 3]), 0)

    def test_tracing_preserves_a_counter(self):
        field = np.zeros((30, 30))
        field[3:27, 3:27] = 1
        field[10:20, 10:20] = 0
        path = renderer.trace(field)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 30 30">'
               f'<path fill="#ffffff" fill-rule="evenodd" d="{path}"/></svg>')
        image = Image.open(BytesIO(renderer.render_svg(svg, 30, 30, 2)))
        self.assertEqual(image.getpixel((30, 30))[3], 0)
        self.assertEqual(image.getpixel((12, 12))[3], 255)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / 'inputs'
            inputs.mkdir()
            source = inputs / 'reference.png'
            Image.new('RGB', (8, 8), 'white').save(source)
            output = root / 'existing'
            output.mkdir()
            sentinel = output / 'keep.txt'
            sentinel.write_text('untouched')
            with self.assertRaises(FileExistsError):
                renderer.rebuild(source, output)
            self.assertEqual(sentinel.read_text(), 'untouched')

    def test_source_directory_is_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'reference.png'
            Image.new('RGB', (8, 8), 'white').save(source)
            with self.assertRaises(ValueError):
                renderer.rebuild(source, root / 'generated')
            self.assertFalse((root / 'generated').exists())

    def test_scaled_or_wrong_input_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / 'inputs'
            inputs.mkdir()
            source = inputs / 'reference.png'
            Image.new('RGB', (8, 8), 'white').save(source)
            output = root / 'generated'
            with self.assertRaises(ValueError):
                renderer.rebuild(source, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
