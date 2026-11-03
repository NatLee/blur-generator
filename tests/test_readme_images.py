"""scripts/make_readme_images.py makes the README pictures by running the commands the README prints.
The commands are written in both places, so check that they still match."""
import pathlib
import runpy
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = runpy.run_path(str(ROOT / 'scripts' / 'make_readme_images.py'), run_name='make_readme_images')


class TestReadmeImageCommands(unittest.TestCase):

    def test_every_picture_comes_from_a_command_printed_in_the_readme(self):
        readme = ' '.join((ROOT / 'README.md').read_text(encoding='utf-8').split())
        for picture, command in {**SCRIPT['GALLERY'], **SCRIPT['DEPTH']}.items():
            with self.subTest(picture=picture):
                self.assertIn(' '.join(command.split()), readme)


if __name__ == '__main__':
    unittest.main()
