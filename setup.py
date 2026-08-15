import os
import sys
import tempfile

from pybind11.setup_helpers import Pybind11Extension
from pybind11.setup_helpers import build_ext as _pybind11_build_ext
from setuptools import setup

_OPENMP_TEST_SRC = """
#include <omp.h>
int main() { return omp_get_max_threads() > 0 ? 0 : 1; }
"""


def _openmp_candidate_flags():
    """(compile_args, link_args) to TRY for OpenMP on this platform.
    Not a guarantee it works -- _openmp_supported() below actually
    compiles and links a trivial program with these flags before
    anything is trusted.
    """
    if sys.platform == "win32":
        return ["/openmp"], []
    if sys.platform == "darwin":
        # Apple Clang doesn't bundle OpenMP; Homebrew's libomp is the
        # standard way to get it. Check a few common install locations
        # rather than assume one -- if none has the header, OpenMP is
        # skipped for this build (see _openmp_supported).
        for prefix in (
            os.environ.get("HOMEBREW_PREFIX"),
            "/opt/homebrew/opt/libomp",
            "/usr/local/opt/libomp",
        ):
            if prefix and os.path.exists(os.path.join(prefix, "include", "omp.h")):
                return (
                    ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
                    [f"-L{prefix}/lib", "-lomp"],
                )
        return [], []
    # Linux and other Unix-likes: gcc/clang both accept -fopenmp directly.
    return ["-fopenmp"], ["-fopenmp"]


def _openmp_supported(compiler):
    """Actually compiles and links a trivial OpenMP program with the
    candidate flags -- verified, not assumed from the platform alone.
    A failed/uncertain result falls back to a sequential build rather
    than risk a broken one: the #pragma omp directives in the C++ source
    are silently ignored (a no-op, not a compile error) wherever OpenMP
    isn't actually enabled, so skipping it here costs parallelism, not
    correctness.

    BSCPP_SKIP_OPENMP=1 forces this to report unsupported regardless of
    what's actually on the build host -- specifically for cibuildwheel's
    macOS wheel builds (see pyproject.toml's [tool.cibuildwheel.macos]):
    a locally-built Homebrew libomp bundled into a wheel via delocate can
    target a NEWER macOS minimum than the wheel itself claims (observed:
    libomp targeting 15.0 inside a wheel declared compatible with 11.0,
    which delocate correctly refuses to ship) -- caught by actually
    running cibuildwheel locally before trusting this config, not
    assumed to work from the YAML alone. Whether a fresh CI runner even
    has libomp installed is host-state this project doesn't control cross-
    OS, so wheel builds pin the answer explicitly instead of depending on
    it. `pip install` from source is unaffected -- OpenMP still gets
    detected and used there exactly as before.
    """
    if os.environ.get("BSCPP_SKIP_OPENMP") == "1":
        return False, [], []
    compile_args, link_args = _openmp_candidate_flags()
    if not compile_args:
        return False, [], []

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "openmp_test.cpp")
        with open(src_path, "w") as f:
            f.write(_OPENMP_TEST_SRC)
        try:
            objects = compiler.compile([src_path], output_dir=tmpdir, extra_postargs=compile_args)
            compiler.link_executable(objects, "openmp_test", output_dir=tmpdir,
                                      extra_postargs=link_args)
            return True, compile_args, link_args
        except Exception:
            return False, [], []


class build_ext(_pybind11_build_ext):
    """Adds OpenMP flags to every extension IF a real compile+link check
    confirms they work on this compiler/platform -- see _openmp_supported.
    """

    def build_extensions(self):
        supported, compile_args, link_args = _openmp_supported(self.compiler)
        if supported:
            print(f"OpenMP: enabled ({' '.join(compile_args)})")
            for ext in self.extensions:
                ext.extra_compile_args = list(ext.extra_compile_args or []) + compile_args
                ext.extra_link_args = list(ext.extra_link_args or []) + link_args
        else:
            print("OpenMP: not available for this compiler/platform -- building without it "
                  "(path loops still run correctly, just single-threaded).")
        super().build_extensions()


ext_modules = [
    Pybind11Extension(
        "bscpp._core",
        sources=[
            "cpp/src/bindings.cpp",
            "cpp/src/black_scholes.cpp",
            "cpp/src/monte_carlo.cpp",
            "cpp/src/longstaff_schwartz.cpp",
            "cpp/src/heston.cpp",
            "cpp/src/crr_tree.cpp",
        ],
        include_dirs=["cpp/include"],
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
