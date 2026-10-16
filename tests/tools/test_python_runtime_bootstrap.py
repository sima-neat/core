#!/usr/bin/env python3
"""Exercise Python's installed Runtime discovery without importing the ARM extension."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "python/pyneat/__init__.py"


class RuntimeBootstrapTest(unittest.TestCase):
    def configure(self, directories, environment=None):
        # Execute the actual bootstrap helpers, not a copied implementation.
        names = {"_prepend_env_path", "_configure_neat_runtime_environment"}
        tree = ast.parse(BOOTSTRAP.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in functions}, names)
        env = dict(environment or {})
        scope = {"_Path": Path, "_os": SimpleNamespace(name="posix", environ=env),
                 "_existing_library_dirs": lambda: directories}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(BOOTSTRAP), "exec"), scope)
        scope["_configure_neat_runtime_environment"]()
        return env

    def test_installed_runtime_without_old_cvu_module(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, plugins = root / "runtime", root / "gst-plugins"
            runtime.mkdir()
            plugins.mkdir()
            (runtime / "libsimaai_runtime_gstreamer.so").touch()
            (plugins / "libgstneatkernelrunner.so").touch()
            env = self.configure([runtime, plugins], {"GST_PLUGIN_PATH": "/custom"})
            self.assertEqual(env["LD_LIBRARY_PATH"].split(":"), [str(runtime), str(plugins)])
            self.assertEqual(env["GST_PLUGIN_PATH"].split(":"), [str(plugins), "/custom"])
            self.assertEqual(env["GST_PLUGIN_PATH_1_0"], str(plugins))
            self.assertFalse((plugins / "libgstneatprocesscvu.so").exists())

    def test_old_module_alone_is_not_selected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "libgstneatprocesscvu.so").touch()
            (root / "libprocesscvu_testhooks.so").touch()
            self.assertEqual(self.configure([root]), {})


if __name__ == "__main__":
    unittest.main()
