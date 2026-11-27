from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get
import os

required_conan_version = ">=2.0.0"


class CmsisNNConan(ConanFile):
    name = "cmsis-nn"
    description = ("CMSIS-NN: optimized neural network kernels for Arm Cortex-M "
                   "processors (convolutions, depthwise, fully connected, pooling, "
                   "activations, LSTM, and the int8/int16 fixed-point helpers)")
    license = "Apache-2.0"
    url = "https://github.com/ARM-software/CMSIS-NN"
    homepage = "https://github.com/ARM-software/CMSIS-NN"
    topics = ("neural-network", "cortex-m", "dsp", "arm", "inference", "kernel")
    package_type = "static-library"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "fPIC": [True, False],
        "f32": [True, False],
        "f16": [True, False],
    }
    default_options = {
        "fPIC": True,
        "f32": False,
        "f16": False,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        # A kernel library has no C++ surface of its own; the sources are C.
        self.settings.rm_safe("compiler.libcxx")
        self.settings.rm_safe("compiler.cppstd")

    def layout(self):
        cmake_layout(self)

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

    def generate(self):
        tc = CMakeToolchain(self)
        # The Python bindings are a helper for generating test data, not part of the
        # library, and they would drag pybind11 and a host interpreter into every build.
        tc.cache_variables["CMSISNN_BUILD_PYBIND"] = False
        tc.cache_variables["ARM_NN_ENABLE_F32"] = bool(self.options.f32)
        tc.cache_variables["ARM_NN_ENABLE_F16"] = bool(self.options.f16)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "LICENSE", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))
        # Upstream has no install rules for the library itself -- the only `install()`
        # in its CMakeLists is inside the pybind block -- so packaging is explicit here.
        # Measured: calling cmake.install() installs nothing, and the consumer then fails
        # with "Library cmsis-nn not found in package".
        copy(self, "*.h", src=os.path.join(self.source_folder, "Include"),
             dst=os.path.join(self.package_folder, "include"))
        copy(self, "*.a", src=self.build_folder,
             dst=os.path.join(self.package_folder, "lib"), keep_path=False)
        copy(self, "*.lib", src=self.build_folder,
             dst=os.path.join(self.package_folder, "lib"), keep_path=False)

    def package_info(self):
        # The consumer's scaffold calls find_package(<metadata key>), and a Conan
        # reference name is lowercase, so the config file answers to `cmsis-nn` while the
        # target keeps upstream's canonical spelling.
        self.cpp_info.set_property("cmake_file_name", "cmsis-nn")
        self.cpp_info.set_property("cmake_target_name", "cmsis-nn::cmsis-nn")
        self.cpp_info.libs = ["cmsis-nn"]
        if self.options.f32:
            self.cpp_info.defines.append("ARM_NN_ENABLE_F32")
        if self.options.f16:
            self.cpp_info.defines.append("ARM_NN_ENABLE_F16")
