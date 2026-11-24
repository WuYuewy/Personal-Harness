# Third-party notices

Personal Harness is distributed under Apache-2.0. Its dependencies retain their own licenses.

The UI includes these vendored browser resources so it can load without third-party CDNs:

- Chart.js 4.5.1 — MIT, license in `src/personal_harness/ui/vendor/Chart.js.LICENSE.md`.
- @kurkle/color — MIT, license in `src/personal_harness/ui/vendor/kurkle-color.LICENSE.md`.
- KaTeX 0.16.22 stylesheet and fonts — MIT, license in `src/personal_harness/ui/vendor/katex/LICENSE`. Upstream: https://github.com/KaTeX/KaTeX .

At startup the app prepares a local entry point from the installed Chainlit frontend; its immutable assets are reused or copied into the private runtime directory. The installed dependency is not modified. Chainlit and its bundled dependencies retain their upstream licenses.

The private frontend copy also adds a null-list guard to cmdk's compiled selector in Chainlit 2.12. This prevents tool selection from crashing when a suggestion list unmounts. The modification is applied by `ui_bootstrap.py`; patched assets are copied, never linked to the installed dependency.
