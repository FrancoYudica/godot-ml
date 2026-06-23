# Development Setup

## IDE - VS Code + clangd

Install the **clangd** extension (`llvm-vs-code-extensions.vscode-clangd`).

Generate the compilation database so clangd understands include paths and flags:

```bash
cd src
scons compiledb platform=windows target=template_debug
```

Re-run this whenever the build configuration changes (new sources, new flags, etc.). The output `src/compile_commands.json` is gitignored.

If the Microsoft C/C++ IntelliSense engine is also installed, disable it in `.vscode/settings.json` to avoid conflicts:

```json
{
  "C_Cpp.intelliSenseEngine": "disabled"
}
```
