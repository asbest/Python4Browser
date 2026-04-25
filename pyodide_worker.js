importScripts('https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.js');

let pyodide;
let llmEngine = null;
let appSettings = {
    useCPU: false
};

function applyGpuPatch() {
    if (self.navigator && self.navigator.gpu) {
        const originalRequestAdapter = self.navigator.gpu.requestAdapter.bind(self.navigator.gpu);
        self.navigator.gpu.requestAdapter = async function(options) {
            let newOptions;
            if (appSettings.useCPU) {
                newOptions = Object.assign({}, options, { forceFallbackAdapter: true });
            } else {
                newOptions = Object.assign({}, options, { powerPreference: "high-performance" });
            }
            let adapter = await originalRequestAdapter(newOptions);
            if (!adapter && appSettings.useCPU) {
                adapter = await originalRequestAdapter(Object.assign({}, options, { powerPreference: "low-power" }));
            }
            return adapter;
        };
    }
}

applyGpuPatch();

// Helper to dynamically load WebLLM from ES Module
async function getWebLLM() {
    return await import("https://esm.run/@mlc-ai/web-llm");
}

self.onmessage = async (e) => {
    const { id, type, payload } = e.data;

    try {
        switch (type) {
            case 'init':
                if (payload.useCPU !== undefined) {
                    appSettings.useCPU = payload.useCPU;
                }
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

                const packages = ['numpy', 'pandas', 'networkx', 'matplotlib', 'micropip', 'jedi'];
                for (let i = 0; i < packages.length; i++) {
                    self.postMessage({ type: 'stdout', text: `Loading package ${i+1}/${packages.length}: ${packages[i]}...\n` });
                    await pyodide.loadPackage(packages[i]);
                }

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

                    # Expose a Python module to interact with WebLLM
                    class BrowserLLM:
                        @staticmethod
                        def load_model(model_id="Llama-3.2-1B-Instruct-q4f16_1-MLC"):
                            import js
                            print(f"Loading LLM {model_id} via WebGPU (this may take a while)...")
                            # We can use js.postMessage to communicate with the worker itself, but
                            # since we are IN the worker, we can directly call JS functions exposed to global.
                            # Better approach: We dispatch a command to our own worker message loop
                            # or just execute it directly if we map a JS async function to global.
                            pass

                    sys.modules['browser_llm'] = BrowserLLM()

                    print("Pyodide initialized.")
                    print(f"Python version: {sys.version.split(' ')[0]}")
                    print("Available libraries: pandas (pd), numpy (np), networkx (nx), matplotlib.pyplot (plt)")
                    print("Added 'browser_llm' to access local LLM. Try: import browser_llm; browser_llm.chat('Hello')")
                    print("Use the 'Upload Files' button to make local files available.")
                    print("--------------------------------------------------")
                `);

                // Expose JS functions directly to the global worker scope so 'import js' can access them
                self._js_load_llm = async (modelId) => {
                    if (llmEngine) {
                        return "Model already loaded.";
                    }
                    try {
                        const webllm = await getWebLLM();
                        const initProgressCallback = (progress) => {
                            // Can print progress to python stdout
                            // console.log(progress);
                        };
                        const cpuLabel = appSettings.useCPU ? " via WebGPU (CPU mode)" : " via WebGPU (High Performance)";
                        console.log(`Loading LLM ${modelId}${cpuLabel}...`);

                        llmEngine = await webllm.CreateMLCEngine(modelId, { initProgressCallback });
                        return `LLM successfully loaded!${appSettings.useCPU ? " [CPU Mode]" : ""}`;
                    } catch (err) {
                        return "Error loading LLM: " + err.message;
                    }
                };

                self._js_ask_llm = async (prompt) => {
                    if (!llmEngine) {
                        throw new Error("LLM is not loaded. Call load_model() first.");
                    }
                    const messages = [{ role: "user", content: prompt }];
                    const reply = await llmEngine.chat.completions.create({ messages });
                    return reply.choices[0].message.content;
                };

                await pyodide.runPythonAsync(`
                    import browser_llm
                    import asyncio

                    async def async_load(model_id="Llama-3.2-1B-Instruct-q4f16_1-MLC"):
                        import js
                        print(f"Loading {model_id}...")
                        result = await js._js_load_llm(model_id)
                        print(result)

                    async def async_chat(prompt):
                        import js
                        try:
                            result = await js._js_ask_llm(prompt)
                            return result
                        except Exception as e:
                            print(e)
                            return str(e)

                    browser_llm.load_model = async_load
                    browser_llm.chat = async_chat
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

            case 'updateSettings':
                appSettings = Object.assign(appSettings, payload);
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
