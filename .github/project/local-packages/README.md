# local-packages

Recipes for a Conan package **no remote carries**, kept here so that git carries them.

Each package is a CCI-shaped recipe:
`.github/project/local-packages/<name>/{config.yml, all/{conandata.yml, conanfile.py, test_package/}}`

Why a recipe and not a built artifact: a recipe builds **from source on whatever machine asks for
it**, so Linux, macOS, Windows and every cross profile get a package built for their own
configuration. A committed binary would be bound to the configuration it was built in -- the
package id carries os, arch, compiler and build_type -- and a foreign host would either skip it or,
worse, link the wrong object format.

What this costs: `conandata.yml` carries the upstream download URL and sha256, which are now part
of the version library. CCI itself does exactly this; the alternative (keeping the recipe out of
tree) is what made every CI platform but one unable to build the leg that needs it.

Getting a package into the cache:
`python3 .github/project/package_sources.py` (no arguments). `--check` only reports.

Adding one: copy the CCI-shaped folder here and add an entry to
`.github/project/package-sources.json`.
