importScripts('https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.js');

let pyodide;

self.onmessage = async (e) => {
    const { id, type, payload } = e.data;

    try {
        switch (type) {
            case 'init':
                pyodide = await loadPyodide();

                // Mount IndexedDB
                const dir = '/workspace';
                pyodide.FS.mkdir(dir);
                pyodide.FS.mount(pyodide.FS.filesystems.IDBFS, {}, dir);

                // Sync IDB into memory
                await new Promise((resolve, reject) => {
                    pyodide.FS.syncfs(true, function (err) {
                        if (err) reject(err);
                        else resolve();
                    });
                });

                pyodide.FS.chdir(dir);

                await pyodide.loadPackage(['numpy', 'pandas', 'networkx', 'matplotlib', 'micropip', 'jedi']);
                await pyodide.runPythonAsync(`
                    import pandas as pd
                    import numpy as np
                    import jedi
                    import networkx as nx
                    import matplotlib.pyplot as plt
                    import io
                    import sys
                    import base64

                    # Configure matplotlib to use the 'Agg' backend
                    import matplotlib
                    matplotlib.use('Agg')

                    # Patch matplotlib.pyplot.show to output a base64 encoded image
                    def _show_patch(*args, **kwargs):
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        img_str = base64.b64encode(buf.read()).decode('utf-8')

                        import js
                        # Post a special message via self.postMessage (js.postMessage maps to the worker global scope)
                        js.postMessage(js.Object.fromEntries(js.Map.new([["type", "plot"], ["data", img_str]])))
                        plt.clf()

                    plt.show = _show_patch

                    print("Pyodide initialized.")
                    print(f"Python version: {sys.version.split(' ')[0]}")
                    print("Available libraries: pandas (pd), numpy (np), networkx (nx), matplotlib.pyplot (plt)")
                    print("Use the 'Upload Files' button to make local files available.")
                    print("--------------------------------------------------")
                `);
                self.postMessage({ id, status: 'success' });
                break;

            case 'run':
                pyodide.setStdout({ batched: (text) => self.postMessage({ type: 'stdout', text }) });
                pyodide.setStderr({ batched: (text) => self.postMessage({ type: 'stderr', text }) });

                let result = await pyodide.runPythonAsync(payload.code);
                if (result !== undefined) {
                    let repr = pyodide.globals.get('repr');
                    self.postMessage({ type: 'repr', text: repr(result) });
                }
                self.postMessage({ id, status: 'success' });
                break;

            case 'readdir':
                const files = pyodide.FS.readdir(payload.path);
                self.postMessage({ id, status: 'success', data: files });
                break;

            case 'readFile':
                let content;
                if (payload.encoding === 'binary') {
                    content = pyodide.FS.readFile(payload.path, { encoding: 'binary' });
                } else {
                    content = pyodide.FS.readFile(payload.path, { encoding: 'utf8' });
                }
                self.postMessage({ id, status: 'success', data: content });
                break;

            case 'writeFile':
                pyodide.FS.writeFile(payload.path, payload.data, { encoding: payload.encoding || 'utf8' });
                await new Promise((resolve, reject) => {
                    pyodide.FS.syncfs(false, err => err ? reject(err) : resolve());
                });
                self.postMessage({ id, status: 'success' });
                break;

            case 'unlink':
                pyodide.FS.unlink(payload.path);
                await new Promise((resolve, reject) => {
                    pyodide.FS.syncfs(false, err => err ? reject(err) : resolve());
                });
                self.postMessage({ id, status: 'success' });
                break;

            case 'unpackArchive':
                pyodide.unpackArchive(payload.data, payload.format);
                await new Promise((resolve, reject) => {
                    pyodide.FS.syncfs(false, err => err ? reject(err) : resolve());
                });
                self.postMessage({ id, status: 'success' });
                break;

            case 'zipExport':
                await pyodide.runPythonAsync(`
                    import shutil
                    import os
                    # Create zip archive of /workspace into /tmp/workspace.zip
                    shutil.make_archive('/tmp/workspace', 'zip', '/workspace')
                `);
                const zipData = pyodide.FS.readFile('/tmp/workspace.zip', { encoding: 'binary' });
                pyodide.FS.unlink('/tmp/workspace.zip');
                self.postMessage({ id, status: 'success', data: zipData });
                break;

            case 'installPackage':
                await pyodide.runPythonAsync(`
                    import micropip
                    await micropip.install('${payload.packageName}')
                `);
                pyodide.pyimport(payload.packageName);
                self.postMessage({ id, status: 'success' });
                break;

            case 'autocomplete':
                try {
                    pyodide.globals.set('_autocomplete_code', payload.code);
                    pyodide.globals.set('_autocomplete_line', payload.line);
                    pyodide.globals.set('_autocomplete_column', payload.column);

                    const completionsJson = await pyodide.runPythonAsync(`
                        import json
                        script = jedi.Script(_autocomplete_code)
                        completions = script.complete(_autocomplete_line, _autocomplete_column)

                        def get_kind(c):
                            t = c.type
                            if t == 'class': return 0 # monaco.languages.CompletionItemKind.Class
                            if t == 'function': return 1 # Function
                            if t == 'keyword': return 13 # Keyword
                            if t == 'module': return 8 # Module
                            if t == 'property': return 9 # Property
                            return 5 # Variable

                        json.dumps([{
                            'label': c.name,
                            'insertText': c.name,
                            'documentation': c.docstring(),
                            'kind': get_kind(c)
                        } for c in completions])
                    `);

                    self.postMessage({ id, status: 'success', data: JSON.parse(completionsJson) });
                } finally {
                    pyodide.globals.delete('_autocomplete_code');
                    pyodide.globals.delete('_autocomplete_line');
                    pyodide.globals.delete('_autocomplete_column');
                }
                break;

            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (err) {
        self.postMessage({ id, status: 'error', error: err.message });
    }
};
