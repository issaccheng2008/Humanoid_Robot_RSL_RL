"""Runtime detection must support Isaac Sim's standalone HPC installation."""
import importlib.metadata
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class RuntimeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'hpc/runtime_compat.py'
        spec = importlib.util.spec_from_file_location('runtime_under_test', path)
        self.runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runtime)

    def detector(self):
        self.assertTrue(hasattr(self.runtime, 'get_isaacsim_version'),
                        'standalone Isaac Sim version detection is missing')
        return self.runtime.get_isaacsim_version

    def test_pip_install_keeps_metadata_version(self):
        with patch.object(self.runtime.importlib.metadata, 'version', return_value='6.0.1.0'):
            version, source = self.detector()()
        self.assertEqual(version, '6.0.1.0')
        self.assertIn('metadata', source)

    def test_binary_install_reads_version_without_package_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'python_packages/isaacsim'
            package.mkdir(parents=True)
            version_file = root / 'VERSION'
            version_file.write_text('6.0.1-rc.7+release.42383\n', encoding='utf-8')
            spec = SimpleNamespace(origin=str(package / '__init__.py'),
                                   submodule_search_locations=[str(package)])
            with patch.object(self.runtime.importlib.metadata, 'version',
                              side_effect=importlib.metadata.PackageNotFoundError('isaacsim')):
                with patch('importlib.util.find_spec', return_value=spec):
                    version, source = self.detector()()
            self.assertEqual(version, '6.0.1-rc.7+release.42383')
            self.assertEqual(source, str(version_file))

    def test_isaac_path_version_still_rejects_wrong_generation(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / 'VERSION').write_text('5.1.0\n', encoding='utf-8')
            with patch.object(self.runtime.importlib.metadata, 'version',
                              side_effect=importlib.metadata.PackageNotFoundError('isaacsim')):
                with patch('importlib.util.find_spec', return_value=None):
                    with patch.dict(os.environ, {'ISAAC_PATH': directory}, clear=True):
                        version, _ = self.detector()()
            with self.assertRaisesRegex(RuntimeError, 'isaacsim.*does not match'):
                self.runtime.validate_versions({'python': '3.12.10', 'isaacsim': version,
                                                'isaaclab_framework': '3.0.0', 'rsl_rl': '5.4.1'})

    def test_no_version_reports_actionable_error(self):
        with patch.object(self.runtime.importlib.metadata, 'version',
                          side_effect=importlib.metadata.PackageNotFoundError('isaacsim')):
            with patch('importlib.util.find_spec', return_value=None):
                with patch.dict(os.environ, {}, clear=True):
                    with patch.object(Path, 'is_file', return_value=False):
                        with self.assertRaisesRegex(RuntimeError, 'VERSION'):
                            self.detector()()

    def test_full_runtime_check_works_when_only_isaacsim_metadata_is_missing(self):
        original = self.runtime.importlib.metadata.version

        def metadata_version(name):
            if name == 'isaacsim':
                raise importlib.metadata.PackageNotFoundError(name)
            return original(name)

        with patch.object(self.runtime.importlib.metadata, 'version', side_effect=metadata_version):
            try:
                runtime = self.runtime.check_runtime()
            except importlib.metadata.PackageNotFoundError as error:
                self.fail(f'check_runtime still requires isaacsim pip metadata: {error}')
        self.assertTrue(runtime['isaacsim'].startswith('6.0.'))
        self.assertTrue(runtime['isaacsim_version_source'].endswith('VERSION'))


if __name__ == '__main__':
    unittest.main()
