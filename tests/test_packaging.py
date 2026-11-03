"""PyPI shows the README without the repository around it, so setup.py turns
every relative link in it into an absolute GitHub link, pinned to the release
tag. The README itself only uses relative links to this repository."""
import pathlib
import re
import runpy
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETUP = runpy.run_path(str(ROOT / 'setup.py'), run_name='setup_under_test')
absolute_links = SETUP['absolute_links']

RAW = 'https://raw.githubusercontent.com/NatLee/blur-generator/v9.9.9/'
PAGE = 'https://github.com/NatLee/blur-generator/blob/v9.9.9/'


class TestReadmeLinks(unittest.TestCase):

    def test_embedded_images_point_at_the_file_in_the_release_tag(self):
        readme = ('| ![photo](doc/readme/sample.jpg) | ![blur](doc/readme/gallery-lens.jpg) |\n'
                  '<img src="doc/readme/strip-depth.png"> <source srcset="doc/a.png">\n')
        self.assertEqual(absolute_links(readme, 'v9.9.9'), (
            '| ![photo](' + RAW + 'doc/readme/sample.jpg) | ![blur](' + RAW + 'doc/readme/gallery-lens.jpg) |\n'
            '<img src="' + RAW + 'doc/readme/strip-depth.png"> <source srcset="' + RAW + 'doc/a.png">\n'))

    def test_links_point_at_the_github_page_in_the_release_tag(self):
        readme = 'The images are in [`doc/`](doc/). [MIT](LICENSE) <a href="scripts/x.py">x</a>'
        self.assertEqual(absolute_links(readme, 'v9.9.9'), (
            'The images are in [`doc/`](' + PAGE + 'doc/). [MIT](' + PAGE + 'LICENSE) '
            '<a href="' + PAGE + 'scripts/x.py">x</a>'))

    def test_a_link_around_an_image_is_rewritten_too(self):
        readme = '[![License](https://img.shields.io/pypi/l/BlurGenerator.svg)](LICENSE)'
        self.assertEqual(absolute_links(readme, 'v9.9.9'),
                         '[![License](https://img.shields.io/pypi/l/BlurGenerator.svg)](' + PAGE + 'LICENSE)')

    def test_absolute_links_and_anchors_are_left_alone(self):
        readme = ('[issue #5](https://github.com/NatLee/blur-generator/issues/5) [up](#installation) '
                  '<img src="https://avatars.githubusercontent.com/u/1"> [mail](mailto:a@b.c) '
                  '<a href="/NatLee">x</a> `cv2.imread(\'./doc/test.png\')`')
        self.assertEqual(absolute_links(readme, 'v9.9.9'), readme)

    def test_the_long_description_links_to_the_tag_of_this_version(self):
        version = SETUP['VERSION']
        description = SETUP['LONG_DESCRIPTION']
        self.assertIn('raw.githubusercontent.com/NatLee/blur-generator/v{}/doc/readme/sample.jpg'.format(version),
                      description)
        self.assertEqual(absolute_links(description, 'v' + version), description)

    def test_the_readme_links_to_its_own_files_relatively(self):
        # Absolute links to this repository's files would bypass the rewrite above,
        # and would keep pointing at whatever branch they name.
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertEqual(
            re.findall(r'github(?:usercontent)?\.com/NatLee/blur-generator/(?:blob|tree|raw|main)/', readme), [])


if __name__ == '__main__':
    unittest.main()
