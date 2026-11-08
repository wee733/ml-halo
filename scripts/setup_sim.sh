#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    cat <<'EOF'
Usage: bash scripts/setup_sim.sh [--jobs N]
Build the local MuJoCo/ONNX Runtime tracker using the existing workstation Python.
Downloads Ubuntu development packages and matching ONNX Runtime headers into .local/.
The C++ application is built under .local/build/sim. Maximum parallel jobs is four.
EOF
    exit 0
fi
jobs=4
if [[ $# == 2 && "$1" == --jobs && "$2" =~ ^[1-4]$ ]]; then
    jobs="$2"
elif [[ $# != 0 ]]; then
    echo "Use --help or --jobs with a value from 1 to 4." >&2
    exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
command -v cmake >/dev/null
command -v apt-get >/dev/null
command -v dpkg-deb >/dev/null

"$HALO_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import urllib.request

if platform.machine() != 'x86_64':
    raise SystemExit('This simulation build profile targets the x86_64 workstation.')
local = Path(os.environ['HALO_LOCAL_ROOT']).resolve()
vendor = local / 'vendor/sim'
sysroot = vendor / 'sysroot'
vendor.mkdir(parents=True, exist_ok=True)
packages = ('ninja-build', 'protobuf-compiler', 'libprotobuf-dev', 'libprotoc32t64',
            'libprotobuf32t64', 'libprotobuf-lite32t64', 'libzmq3-dev', 'libzmq5',
            'libglfw3-dev', 'libglfw3')
lock = vendor / 'ubuntu-packages.json'
if lock.is_file():
    versions = json.loads(lock.read_text())
else:
    versions = {}
    for package in packages:
        policy = subprocess.check_output(['apt-cache', 'policy', package], text=True)
        match = re.search(r'^\s*Candidate:\s*(\S+)', policy, re.MULTILINE)
        if not match or match.group(1) == '(none)':
            raise SystemExit(f'Ubuntu package unavailable in the configured repositories: {package}')
        versions[package] = match.group(1)
    lock.write_text(json.dumps(versions, indent=2) + '\n')
for package in packages:
    download = vendor / 'downloads' / package / versions[package]
    download.mkdir(parents=True, exist_ok=True)
    archives = list(download.glob('*.deb'))
    if not archives:
        subprocess.run(['apt-get', 'download', f'{package}={versions[package]}'],
                       cwd=download, check=True)
        archives = list(download.glob('*.deb'))
    if len(archives) != 1:
        raise SystemExit(f'Expected one package archive in {download}')
    subprocess.run(['dpkg-deb', '-x', str(archives[0]), str(sysroot)], check=True)

import mujoco
import onnxruntime

if mujoco.__version__ != '3.2.5':
    raise SystemExit('The simulation build requires the project-local mujoco==3.2.5 overlay.')
mujoco_root = Path(mujoco.__file__).parent
mujoco_lib = mujoco_root / 'libmujoco.so.3.2.5'
ort_root = Path(onnxruntime.__file__).parent
ort_lib = ort_root / 'capi' / f'libonnxruntime.so.{onnxruntime.__version__}'
if not mujoco_lib.is_file() or not ort_lib.is_file():
    raise SystemExit('The selected Python wheels do not expose the required native shared libraries.')

ort_sdk = vendor / f'onnxruntime-{onnxruntime.__version__}'
ort_link_dir = ort_sdk / 'lib'
ort_link_dir.mkdir(parents=True, exist_ok=True)
for name in ('libonnxruntime.so', 'libonnxruntime.so.1'):
    link = ort_link_dir / name
    link.unlink(missing_ok=True)
    link.symlink_to(ort_lib)
include = ort_sdk / 'include'
include.mkdir(parents=True, exist_ok=True)
source = ('https://raw.githubusercontent.com/microsoft/onnxruntime/'
          f'v{onnxruntime.__version__}/include/onnxruntime/core/session/')
pending = ['onnxruntime_cxx_api.h']
seen = set()
while pending:
    name = pending.pop()
    if name in seen:
        continue
    seen.add(name)
    if Path(name).name != name:
        raise SystemExit(f'Unexpected ONNX Runtime header reference: {name}')
    header = include / name
    if not header.is_file():
        print(f'Downloading {source}{name}', flush=True)
        with urllib.request.urlopen(source + name, timeout=60) as response:
            content = response.read()
        temporary = header.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(header)
    pending.extend(re.findall(r'^#include\s+"([^"]+)"', header.read_text(), re.MULTILINE))

cmake_dir = vendor / 'mujoco'
cmake_dir.mkdir(parents=True, exist_ok=True)
(cmake_dir / 'mujocoConfig.cmake').write_text(
    'if(NOT TARGET mujoco::mujoco)\n'
    '  add_library(mujoco::mujoco SHARED IMPORTED)\n'
    '  set_target_properties(mujoco::mujoco PROPERTIES\n'
    f'    IMPORTED_LOCATION "{mujoco_lib}"\n'
    f'    INTERFACE_INCLUDE_DIRECTORIES "{mujoco_root / "include"}")\n'
    'endif()\n'
)
cache = vendor / 'dependencies.cmake'
values = {
    'mujoco_DIR': cmake_dir,
    'ONNXRuntime_INCLUDE_DIR': include,
    'ONNXRuntime_LIBRARY': ort_link_dir / 'libonnxruntime.so',
    'Protobuf_INCLUDE_DIR': sysroot / 'usr/include',
    'Protobuf_LIBRARY': sysroot / 'usr/lib/x86_64-linux-gnu/libprotobuf.so',
    'Protobuf_PROTOC_EXECUTABLE': sysroot / 'usr/bin/protoc',
    'ZeroMQ_INCLUDE_DIR': sysroot / 'usr/include',
    'ZeroMQ_LIBRARY': sysroot / 'usr/lib/x86_64-linux-gnu/libzmq.so',
    'glfw3_DIR': sysroot / 'usr/lib/x86_64-linux-gnu/cmake/glfw3',
    'CMAKE_MAKE_PROGRAM': sysroot / 'usr/bin/ninja',
}
cache.write_text(''.join(f'set({key} "{value}" CACHE STRING "Local simulation dependency" FORCE)\n'
                         for key, value in values.items()))
build = local / 'build/sim'
build_cache = build / 'CMakeCache.txt'
if build_cache.is_file():
    cached = build_cache.read_text()
    previous_source = re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$', cached, re.MULTILINE)
    expected_source = Path(os.environ['HALO_ROOT']) / 'system/cpp'
    if previous_source and Path(previous_source.group(1)) != expected_source:
        if 'CMAKE_PROJECT_NAME:STATIC=halo_system\n' not in cached:
            raise SystemExit(f'Unrecognized CMake build at {build}; choose another HALO_LOCAL_ROOT.')
        print(f'Recreating moved simulation build directory {build}', flush=True)
        shutil.rmtree(build)
print(f'Prepared simulation dependencies in {vendor}', flush=True)
PY

sim_vendor="$HALO_LOCAL_ROOT/vendor/sim"
export LD_LIBRARY_PATH="$sim_vendor/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S "$HALO_ROOT/system/cpp" -B "$HALO_LOCAL_ROOT/build/sim" -G Ninja \
    -C "$sim_vendor/dependencies.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DRT_DEPLOY_JETSON_PLATFORM=none -DRT_DEPLOY_WITH_MUJOCO=ON \
    -DRT_DEPLOY_WITH_ORT=ON -DRT_DEPLOY_WITH_TRT=OFF -DRT_DEPLOY_WITH_UNITREE=OFF \
    -DRT_DEPLOY_WITH_REALSENSE=OFF -DRT_DEPLOY_BUILD_TESTS=OFF -DRT_DEPLOY_ENABLE_WERROR=OFF
cmake --build "$HALO_LOCAL_ROOT/build/sim" --target motiontrack_mujoco_zmq -j "$jobs"
echo "Built $HALO_LOCAL_ROOT/build/sim/motiontrack_mujoco_zmq"
