from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "bscpp._core",
        sources=[
            "cpp/src/bindings.cpp",
            "cpp/src/black_scholes.cpp",
            "cpp/src/monte_carlo.cpp",
            "cpp/src/longstaff_schwartz.cpp",
            "cpp/src/heston.cpp",
        ],
        include_dirs=["cpp/include"],
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
