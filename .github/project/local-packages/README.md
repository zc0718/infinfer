# local-packages

Compiled artifacts for a Conan package **no remote carries**, kept here so that git carries them
and CI can build them into the cache.

- One directory per package: `<name>/<version>/` containing `include/`, `lib/`, `licenses/`.
- The `conanfile.py` next to the payload is **generated** by
  `python3 .github/project/package_sources.py` -- do not edit it by hand, and expect it to be
  rewritten.
- The entry that describes the package lives in `.github/project/package-sources.json`.
- Nothing here downloads anything: the upstream recipe, which carries a URL and an sha256, stays
  outside this repository. That is the whole point of keeping the built payload instead.

Removing a package: delete its directory and its entry in `package-sources.json` together --
`deps_gate` compares the two.
