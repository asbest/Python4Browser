# Python4Browser
Interactive Python Console in the Browser
This project is a
fully-featured, interactive Python console (REPL—Read-Eval-Print Loop) that runs entirely in your web browser. It is encapsulated in a single HTML file and requires no server-side backend or local dependencies.
The application leverages Pyodide to execute a complete CPython runtime environment directly in the browser using WebAssembly.
Key Advantages
 * Zero Setup: Absolutely no installation of Python, pip, or virtual environments is required on your local machine. A modern web browser is all you need.
 * Portability: The entire application is a single HTML file. It can be easily shared, emailed, stored on a USB drive, or hosted on any static website.
 * Batteries Included: Powerful scientific libraries, which can often be complex to install, are pre-loaded and ready to use instantly:
   * pandas (for data analysis)
   * numpy (for numerical computing)
   * networkx (for graph theory)
 * Instant Access: Ideal for quick prototyping, teaching or learning Python, or for use in environments where software installation is restricted (e.g., on tablets, Chromebooks, or locked-down corporate devices).
 * Secure (Sandboxed): All code runs within the browser's security sandbox, with no direct access to the user's local file system.
How to Use
 * Save the html file.
 * Open the file in any modern web browser (e.g., Chrome, Firefox, Safari, Edge).
 * Wait a moment for Pyodide and the packages to load in the background (a loading indicator will show the progress).
 * Start typing Python code directly into the console and press Enter to execute.
 * 
