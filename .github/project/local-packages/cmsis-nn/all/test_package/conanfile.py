from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout


class CmsisNNTestConan(ConanFile):
    name = "cmsis-nn-test"
    test_type = "explicit"
    settings = "os", "arch", "compiler", "build_type"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        # CMakeDeps writes the CMSIS-NN config file that find_package reads; without it
        # CMake looks for a config the recipe never installed.
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        # Run the built binary: including a header proves the include path, calling a
        # function proves the archive actually contains the code.
        self.run(self._binary(), env="conanrun")

    def _binary(self):
        import os
        name = "test_package"
        if self.settings.os == "Windows":
            name += ".exe"
        return os.path.join(self.cpp.build.bindirs[0], name)
